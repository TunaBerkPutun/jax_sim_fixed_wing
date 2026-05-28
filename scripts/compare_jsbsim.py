#!/usr/bin/env python3
"""Side-by-side trajectory comparison: JSBSim vs jax_sim.

JSBSim is run as the reference FDM. We trim a chosen aircraft (default c172p)
at a level cruise condition, capture the trim controls and steady state,
then drive both simulators through the *same* control schedule
(trim + an elevator pulse) and overlay the resulting trajectories.

Caveat: c172p is a 1100 kg general-aviation aircraft; jax_sim models a 1 kg
PX4-SIH-style UAV. Numerical agreement is not the goal. What we are looking
for is qualitative agreement in dynamics: stable trim, sane sign conventions,
plausible response shape (rise time / damping) to a stick input. For
quantitative validation you would need a JSBSim XML matched to the
AircraftParams in jax_sim/physics/constants.py.

Outputs:
  jsbsim_log.csv      JSBSim trajectory (1 row per outer step)
  jax_log.csv         jax_sim trajectory (1 row per outer step)
  compare_jsbsim.png  6-panel overlay (altitude, airspeed, pitch, roll, q, p)
"""

from __future__ import annotations

import argparse
import csv

import jax
import jax.numpy as jnp
import jsbsim
import matplotlib.pyplot as plt
import numpy as np

from jax_sim.vehicles.fixed_wing.tier1 import equations_of_motion
from jax_sim.utils.quaternion import quat_to_euler_jax

FT_PER_M = 3.28084
M_PER_FT = 1.0 / FT_PER_M


# ---------------------------------------------------------------------------
# Control schedule (shared by both sims)
# ---------------------------------------------------------------------------

def make_schedule(trim_ail, trim_ele, trim_rud, trim_thr,
                  pulse_t, pulse_dt, pulse_ele, pulse_ail=0.0, pulse_rud=0.0):
    """Return f(t) -> (ail, ele, rud, thr) in JSBSim norm units [-1,1]."""
    def sched(t):
        in_pulse = pulse_t <= t < pulse_t + pulse_dt
        ele = trim_ele + (pulse_ele if in_pulse else 0.0)
        ail = trim_ail + (pulse_ail if in_pulse else 0.0)
        rud = trim_rud + (pulse_rud if in_pulse else 0.0)
        return ail, ele, rud, trim_thr
    return sched


# ---------------------------------------------------------------------------
# JSBSim driver
# ---------------------------------------------------------------------------

def run_jsbsim(aircraft, duration, dt, alt_ft, speed_kts, sched_factory,
               fdm_root=None, trim_overrides=None, alpha_deg=0.0):
    """Run JSBSim trajectory. `trim_overrides` lets callers skip do_trim and
    inject known (ail, ele, rud, thr) trim controls (used for the matched
    aircraft, which has no piston engine and a pre-known trim point)."""
    fdm = jsbsim.FGFDMExec(fdm_root)
    fdm.set_debug_level(0)
    if not fdm.load_model(aircraft):
        raise RuntimeError(f"JSBSim failed to load aircraft '{aircraft}'")
    fdm.set_dt(dt)

    fdm.set_property_value("ic/h-sl-ft", alt_ft)
    fdm.set_property_value("ic/vt-kts", speed_kts)
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    if alpha_deg != 0.0:
        fdm.set_property_value("ic/alpha-deg", alpha_deg)
        fdm.set_property_value("ic/theta-deg", alpha_deg)

    if trim_overrides is None:
        # Stock JSBSim aircraft: piston engine + do_trim
        fdm.set_property_value("fcs/throttle-cmd-norm", 0.5)
        fdm.set_property_value("fcs/mixture-cmd-norm", 1.0)
        fdm.get_propulsion().init_running(-1)
        if not fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")
        try:
            fdm.do_trim(1)
        except RuntimeError as e:
            print(f"[JSBSim] do_trim failed, continuing un-trimmed: {e}")
        trim_ail = fdm.get_property_value("fcs/aileron-cmd-norm")
        trim_ele = fdm.get_property_value("fcs/elevator-cmd-norm")
        trim_rud = fdm.get_property_value("fcs/rudder-cmd-norm")
        trim_thr = fdm.get_property_value("fcs/throttle-cmd-norm")
    else:
        # Custom aircraft: inject trim controls directly and pre-seed lag filters.
        trim_ail, trim_ele, trim_rud, trim_thr = trim_overrides
        fdm.set_property_value("fcs/aileron-cmd-norm", trim_ail)
        fdm.set_property_value("fcs/elevator-cmd-norm", trim_ele)
        fdm.set_property_value("fcs/rudder-cmd-norm", trim_rud)
        fdm.set_property_value("fcs/throttle-cmd-norm", trim_thr)
        # Pre-seed lag filter outputs so we start at steady state.
        for prop, val in [
            ("fcs/throttle-pos-norm", trim_thr),
            ("fcs/elevator-pos-norm", trim_ele),
            ("fcs/left-aileron-pos-norm", trim_ail),
            ("fcs/rudder-pos-norm", trim_rud),
        ]:
            try:
                fdm.set_property_value(prop, val)
            except Exception:
                pass
        if not fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")

    print(f"[JSBSim] trim controls: ail={trim_ail:+.4f} ele={trim_ele:+.4f} "
          f"rud={trim_rud:+.4f} thr={trim_thr:.4f}")

    sched = sched_factory(trim_ail, trim_ele, trim_rud, trim_thr)

    steps = int(round(duration / dt))
    log = np.zeros((steps, 9), dtype=np.float64)  # t, h_m, V_m_s, theta, phi, psi, p, q, r

    for i in range(steps):
        t = i * dt
        ail, ele, rud, thr = sched(t)
        fdm.set_property_value("fcs/aileron-cmd-norm", ail)
        fdm.set_property_value("fcs/elevator-cmd-norm", ele)
        fdm.set_property_value("fcs/rudder-cmd-norm", rud)
        fdm.set_property_value("fcs/throttle-cmd-norm", thr)
        if not fdm.run():
            print(f"[JSBSim] run() halted at step {i} (t={t:.2f}s)")
            log = log[:i]
            break
        log[i, 0] = t
        log[i, 1] = fdm.get_property_value("position/h-sl-ft") * M_PER_FT
        log[i, 2] = fdm.get_property_value("velocities/vt-fps") * M_PER_FT
        log[i, 3] = fdm.get_property_value("attitude/theta-rad")
        log[i, 4] = fdm.get_property_value("attitude/phi-rad")
        log[i, 5] = fdm.get_property_value("attitude/psi-rad")
        log[i, 6] = fdm.get_property_value("velocities/p-rad_sec")
        log[i, 7] = fdm.get_property_value("velocities/q-rad_sec")
        log[i, 8] = fdm.get_property_value("velocities/r-rad_sec")

    return log, (trim_ail, trim_ele, trim_rud, trim_thr)


# ---------------------------------------------------------------------------
# jax_sim driver
# ---------------------------------------------------------------------------

def run_jax_sim(duration, dt, speed_m_s, alt_m, sched_factory, trim_controls,
                alpha_deg=0.0):
    sched = sched_factory(*trim_controls)
    steps = int(round(duration / dt))

    # Match JSBSim's trimmed posture: pitch attitude = alpha (gamma=0 -> theta=alpha),
    # velocity along the *world* horizontal (so ground-track is level).
    a = np.deg2rad(alpha_deg)
    # Body-to-earth quaternion for a pitch rotation about world-y.
    quat = jnp.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    # Steady level cruise: world velocity = [V, 0, 0] (horizontal).
    state0 = jnp.array([
        0.0, 0.0, -alt_m,
        speed_m_s, 0.0, 0.0,
        quat[0], quat[1], quat[2], quat[3],
        0.0, 0.0, 0.0,
        # Pre-seed actuator states with trim controls so motor/servo lag does
        # not introduce a transient at t=0.
        trim_controls[0] * float(np.deg2rad(15.0)),
        trim_controls[1] * float(np.deg2rad(15.0)),
        trim_controls[2] * float(np.deg2rad(15.0)),
        float(trim_controls[3]),
    ])

    ts = np.arange(steps) * dt
    cmds = np.array([sched(t) for t in ts], dtype=np.float32)  # (steps, 4)
    cmds_j = jnp.asarray(cmds)

    def step_fn(s, u):
        s_next = equations_of_motion(s, u, dt)
        return s_next, s_next

    _, traj = jax.lax.scan(step_fn, state0, cmds_j)
    traj = np.asarray(traj)  # (steps, 17)

    pos = traj[:, 0:3]
    vel = traj[:, 3:6]
    quat = traj[:, 6:10]
    omega = traj[:, 10:13]

    # Convert quaternion -> euler (roll, pitch, yaw) using project util.
    euler = np.asarray(jax.vmap(quat_to_euler_jax)(jnp.asarray(quat)))  # (steps, 3)

    h_m = -pos[:, 2]
    v_m_s = np.linalg.norm(vel, axis=1)

    log = np.zeros((steps, 9), dtype=np.float64)
    log[:, 0] = ts
    log[:, 1] = h_m
    log[:, 2] = v_m_s
    log[:, 3] = euler[:, 1]  # pitch (theta)
    log[:, 4] = euler[:, 0]  # roll  (phi)
    log[:, 5] = euler[:, 2]  # yaw   (psi)
    log[:, 6] = omega[:, 0]  # p
    log[:, 7] = omega[:, 1]  # q
    log[:, 8] = omega[:, 2]  # r
    return log


# ---------------------------------------------------------------------------
# Plotting / IO
# ---------------------------------------------------------------------------

COL_NAMES = ["t_s", "h_m", "V_mps", "theta_rad", "phi_rad", "psi_rad",
             "p_radps", "q_radps", "r_radps"]


def save_csv(path, log):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COL_NAMES)
        w.writerows(log.tolist())
    print(f"Wrote {path}  ({len(log)} rows)")


def plot_overlay(jsb, jax_, out_path, pulse_t, pulse_dt):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    panels = [
        ("Altitude [m]",     1, 1.0),
        ("Airspeed [m/s]",   2, 1.0),
        ("Pitch theta [deg]",  3, 180.0 / np.pi),
        ("Roll phi [deg]",     4, 180.0 / np.pi),
        ("Pitch rate q [deg/s]", 7, 180.0 / np.pi),
        ("Roll rate p [deg/s]",  6, 180.0 / np.pi),
    ]
    for ax, (label, col, scale) in zip(axes.flat, panels):
        ax.plot(jsb[:, 0],  jsb[:, col]  * scale, label="JSBSim",  lw=1.4)
        ax.plot(jax_[:, 0], jax_[:, col] * scale, label="jax_sim", lw=1.4, ls="--")
        ax.axvspan(pulse_t, pulse_t + pulse_dt, color="0.85", zorder=0,
                   label="elev pulse")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aircraft", default="px4_sih_uav",
                   help="JSBSim aircraft directory name (default: px4_sih_uav). "
                        "Use 'c172p' for the stock GA reference, or any other "
                        "JSBSim aircraft installed under --fdm-root.")
    p.add_argument("--fdm-root", default="jsbsim_data",
                   help="JSBSim FDM root directory (default: jsbsim_data). "
                        "Set to empty string to use the bundled JSBSim default root.")
    p.add_argument("--matched", action="store_true", default=None,
                   help="Force matched-aircraft trim path (no engine init, "
                        "pre-known trim controls). Default: True if --aircraft "
                        "is px4_sih_uav, else False.")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--dt", type=float, default=0.004,
                   help="Outer-step timestep; matches jax_sim default (250 Hz)")
    p.add_argument("--alt-ft", type=float, default=328.084,
                   help="Trim altitude (ft MSL). Default = 100 m.")
    p.add_argument("--speed-kts", type=float, default=38.88,
                   help="Trim airspeed (kts). Default = 20 m/s.")
    p.add_argument("--pulse-t", type=float, default=5.0)
    p.add_argument("--pulse-dt", type=float, default=0.5)
    p.add_argument("--pulse-ele", type=float, default=-0.15,
                   help="Elevator pulse magnitude (norm), added on top of trim.")
    p.add_argument("--pulse-ail", type=float, default=0.0,
                   help="Aileron pulse magnitude (norm).")
    p.add_argument("--pulse-rud", type=float, default=0.0,
                   help="Rudder pulse magnitude (norm).")
    p.add_argument("--jsbsim-csv", default="jsbsim_log.csv")
    p.add_argument("--jax-csv", default="jax_log.csv")
    p.add_argument("--plot", default="compare_jsbsim.png")
    args = p.parse_args()

    # Auto-enable --matched for our custom aircraft.
    use_matched = args.matched if args.matched is not None else (args.aircraft == "px4_sih_uav")

    sched_factory = lambda a, e, r, t: make_schedule(
        a, e, r, t, args.pulse_t, args.pulse_dt, args.pulse_ele,
        pulse_ail=args.pulse_ail, pulse_rud=args.pulse_rud,
    )

    fdm_root = args.fdm_root if args.fdm_root else None
    trim_overrides = None
    alpha_deg = 0.0
    if use_matched:
        # jax_sim's level-flight trim at V=20 m/s, alt=100 m
        # (from scripts/extract_aero_derivs.py).
        trim_overrides = (0.0, 0.0758, 0.0, 0.3792)
        alpha_deg = -0.843

    print(f"=== JSBSim ({args.aircraft}; root={fdm_root}) ===")
    jsb_log, trim_controls = run_jsbsim(
        args.aircraft, args.duration, args.dt,
        args.alt_ft, args.speed_kts, sched_factory,
        fdm_root=fdm_root,
        trim_overrides=trim_overrides,
        alpha_deg=alpha_deg,
    )
    save_csv(args.jsbsim_csv, jsb_log)

    speed_m_s = args.speed_kts * 0.514444
    alt_m = args.alt_ft * M_PER_FT
    print(f"=== jax_sim (IC: V={speed_m_s:.2f} m/s, h={alt_m:.1f} m, "
          f"alpha={alpha_deg:+.2f} deg) ===")
    jax_log = run_jax_sim(
        args.duration, args.dt, speed_m_s, alt_m,
        sched_factory, trim_controls,
        alpha_deg=alpha_deg,
    )
    save_csv(args.jax_csv, jax_log)

    plot_overlay(jsb_log, jax_log, args.plot, args.pulse_t, args.pulse_dt)


if __name__ == "__main__":
    main()

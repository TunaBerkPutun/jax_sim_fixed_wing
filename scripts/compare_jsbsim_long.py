#!/usr/bin/env python3
"""JSBSim vs jax_sim aero validation suite — independent maneuver tests.

Each test case is run twice (JSBSim, jax_sim) from the *same* matched trim,
with a per-test duration sized so the airframe stays well clear of stall
or ground impact. Tests are completely independent — a crash in one does
not pollute the others.

Rationale: a previous single-batch 180 s schedule drove both sims into a
ground impact during the sustained-aileron segment, after which all
downstream "doublet" and "chirp" metrics ran on diverged state and were
meaningless. Independent tests bound the divergence per maneuver.

Test cases (default suite):
    trim_hold        30 s   open-loop trim drift
    elev_doublet_sm  25 s   small elevator doublet  (linear regime)
    elev_doublet_lg  15 s   larger elevator doublet (mildly nonlinear)
    ail_doublet      15 s   aileron doublet         (roll mode)
    rud_doublet      15 s   rudder doublet          (dutch-roll)
    ail_step_small   10 s   small sustained aileron (drift to bank/spiral)
    throttle_step    20 s   throttle step           (long-period speed)
    elev_chirp       30 s   elevator chirp 0.2->2.0 Hz, A=0.05

Outputs (under --output-dir, default aero_baseline/):
    <case>/jsbsim.csv
    <case>/jax_sim.csv
    <case>/schedule.csv
    <case>/compare.png
    metrics.json               per-case scalar metrics + cross-sim divergence
    summary.txt                human-readable digest

Re-run with `--output-dir aero_v2/` after an aero edit and diff
metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import jsbsim
import matplotlib.pyplot as plt
import numpy as np

from jax_sim.vehicles.fixed_wing.tier1 import equations_of_motion
from jax_sim.utils.quaternion import quat_to_euler_jax

FT_PER_M = 3.28084
M_PER_FT = 1.0 / FT_PER_M
GROUND_GUARD_M = 5.0   # truncate log when altitude drops below this

Trim = Tuple[float, float, float, float]
ScheduleFn = Callable[[float, Trim], Trim]

# ---------------------------------------------------------------------------
# Maneuver primitives
# ---------------------------------------------------------------------------

def hold(t: float, trim: Trim) -> Trim:
    return trim


def make_doublet(axis: str, amp: float, t_on: float = 1.0, dt: float = 0.5) -> ScheduleFn:
    """Two back-to-back pulses on `axis`: +amp then -amp, each of `dt` seconds.

    Axis is one of 'ail', 'ele', 'rud'. Throttle doublets aren't useful.
    """
    idx = {"ail": 0, "ele": 1, "rud": 2}[axis]

    def sched(t: float, trim: Trim) -> Trim:
        out = list(trim)
        if t_on <= t < t_on + dt:
            out[idx] = trim[idx] + amp
        elif t_on + dt <= t < t_on + 2.0 * dt:
            out[idx] = trim[idx] - amp
        return tuple(out)  # type: ignore[return-value]

    return sched


def make_step(axis: str, amp: float, t_on: float = 1.0) -> ScheduleFn:
    """Sustained step on one axis from `t_on` onwards."""
    idx = {"ail": 0, "ele": 1, "rud": 2, "thr": 3}[axis]

    def sched(t: float, trim: Trim) -> Trim:
        out = list(trim)
        if t >= t_on:
            out[idx] = float(np.clip(trim[idx] + amp, -1.0, 1.0))
        return tuple(out)  # type: ignore[return-value]

    return sched


def make_chirp(axis: str, amp: float, f_lo: float, f_hi: float,
               t_on: float = 1.0, t_off: float = 31.0) -> ScheduleFn:
    """Linear-frequency-sweep on `axis` between t_on and t_off."""
    idx = {"ail": 0, "ele": 1, "rud": 2}[axis]
    T = t_off - t_on

    def sched(t: float, trim: Trim) -> Trim:
        out = list(trim)
        if t_on <= t < t_off:
            tau = t - t_on
            phi = 2.0 * np.pi * (f_lo * tau + 0.5 * (f_hi - f_lo) / T * tau * tau)
            out[idx] = trim[idx] + amp * float(np.sin(phi))
        return tuple(out)  # type: ignore[return-value]

    return sched


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    name: str
    duration: float
    sched: ScheduleFn
    description: str


SUITE: List[TestCase] = [
    TestCase("trim_hold",       30.0, hold,
             "open-loop drift from matched trim"),
    TestCase("elev_doublet_sm", 25.0, make_doublet("ele", 0.05),
             "small elevator doublet, free decay (phugoid visible)"),
    TestCase("elev_doublet_lg", 15.0, make_doublet("ele", 0.15),
             "larger elevator doublet, short-period peak"),
    TestCase("ail_doublet",     15.0, make_doublet("ail", 0.30),
             "aileron doublet, roll mode response"),
    TestCase("rud_doublet",     15.0, make_doublet("rud", 0.20),
             "rudder doublet, dutch-roll response"),
    TestCase("ail_step_small",   8.0, make_step("ail", 0.03),
             "small sustained aileron, bank-rate buildup (no pitch compensation)"),
    TestCase("throttle_step",   20.0, make_step("thr", 0.20),
             "throttle step, long-period speed/alt response"),
    TestCase("elev_chirp",      30.0, make_chirp("ele", 0.05, 0.2, 2.0,
                                                 t_on=1.0, t_off=29.0),
             "elevator chirp 0.2 -> 2.0 Hz, A=0.05"),
]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_jsbsim(aircraft: str, fdm_root: str | None, case: TestCase, dt: float,
               alt_ft: float, speed_kts: float, trim: Trim, alpha_deg: float
               ) -> Tuple[np.ndarray, bool]:
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
    fdm.set_property_value("ic/alpha-deg", alpha_deg)
    fdm.set_property_value("ic/theta-deg", alpha_deg)

    a, e, r, th = trim
    fdm.set_property_value("fcs/aileron-cmd-norm", a)
    fdm.set_property_value("fcs/elevator-cmd-norm", e)
    fdm.set_property_value("fcs/rudder-cmd-norm", r)
    fdm.set_property_value("fcs/throttle-cmd-norm", th)
    for prop, val in [
        ("fcs/throttle-pos-norm", th),
        ("fcs/elevator-pos-norm", e),
        ("fcs/left-aileron-pos-norm", a),
        ("fcs/rudder-pos-norm", r),
    ]:
        try:
            fdm.set_property_value(prop, val)
        except Exception:
            pass
    if not fdm.run_ic():
        raise RuntimeError("JSBSim run_ic() failed")

    steps = int(round(case.duration / dt))
    log = np.zeros((steps, 13), dtype=np.float64)
    crashed = False
    last_valid = steps
    for i in range(steps):
        t = i * dt
        ail, ele, rud, thr = case.sched(t, trim)
        fdm.set_property_value("fcs/aileron-cmd-norm", ail)
        fdm.set_property_value("fcs/elevator-cmd-norm", ele)
        fdm.set_property_value("fcs/rudder-cmd-norm", rud)
        fdm.set_property_value("fcs/throttle-cmd-norm", thr)
        if not fdm.run():
            crashed = True
            last_valid = i
            break
        h = fdm.get_property_value("position/h-sl-ft") * M_PER_FT - (alt_ft * M_PER_FT - 100.0)
        # h is altitude above start; flag crash if below floor
        if h < GROUND_GUARD_M - 100.0:  # i.e. < 5 m AGL from a 100 m start
            crashed = True
            last_valid = i
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
        log[i, 9:13] = ail, ele, rud, thr
    return log[:last_valid], crashed


def run_jax_sim(case: TestCase, dt: float, speed_m_s: float, alt_m: float,
                trim: Trim, alpha_deg: float) -> Tuple[np.ndarray, bool]:
    steps = int(round(case.duration / dt))
    a = np.deg2rad(alpha_deg)
    quat = jnp.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    state0 = jnp.array([
        0.0, 0.0, -alt_m,
        speed_m_s, 0.0, 0.0,
        quat[0], quat[1], quat[2], quat[3],
        0.0, 0.0, 0.0,
        trim[0] * float(np.deg2rad(15.0)),
        trim[1] * float(np.deg2rad(15.0)),
        trim[2] * float(np.deg2rad(15.0)),
        float(trim[3]),
    ])
    ts = np.arange(steps) * dt
    cmds = np.array([case.sched(float(t), trim) for t in ts], dtype=np.float32)
    cmds_j = jnp.asarray(cmds)

    def step_fn(s, u):
        s_next = equations_of_motion(s, u, dt)
        return s_next, s_next

    _, traj = jax.lax.scan(step_fn, state0, cmds_j)
    traj = np.asarray(traj)

    pos = traj[:, 0:3]
    vel = traj[:, 3:6]
    quat = traj[:, 6:10]
    omega = traj[:, 10:13]
    euler = np.asarray(jax.vmap(quat_to_euler_jax)(jnp.asarray(quat)))
    h_m = -pos[:, 2]
    V_m_s = np.linalg.norm(vel, axis=1)

    log = np.zeros((steps, 13), dtype=np.float64)
    log[:, 0] = ts
    log[:, 1] = h_m
    log[:, 2] = V_m_s
    log[:, 3] = euler[:, 1]
    log[:, 4] = euler[:, 0]
    log[:, 5] = euler[:, 2]
    log[:, 6:9] = omega
    log[:, 9:13] = cmds

    # Ground-guard: truncate at the first sample below the floor.
    crashed_idx = np.argmax(h_m < GROUND_GUARD_M) if np.any(h_m < GROUND_GUARD_M) else None
    if crashed_idx is not None and crashed_idx > 0:
        return log[:crashed_idx], True
    return log, False


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "abs_max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std":  float(np.std(x)),
        "min":  float(np.min(x)),
        "max":  float(np.max(x)),
        "abs_max": float(np.max(np.abs(x))),
    }


def _peak(t: np.ndarray, sig: np.ndarray) -> dict:
    if len(sig) < 2:
        return {"peak_pos": float("nan"), "t_peak_pos": float("nan"),
                "peak_neg": float("nan"), "t_peak_neg": float("nan")}
    s = sig - sig[0]
    ip = int(np.argmax(s)); ineg = int(np.argmin(s))
    return {
        "peak_pos": float(s[ip]),
        "t_peak_pos": float(t[ip] - t[0]),
        "peak_neg": float(s[ineg]),
        "t_peak_neg": float(t[ineg] - t[0]),
    }


def _zero_crossing_period(t: np.ndarray, sig: np.ndarray) -> float:
    if len(sig) < 4:
        return float("nan")
    s = sig - np.mean(sig)
    zc = np.where(np.diff(np.signbit(s).astype(int)) == -1)[0]
    if len(zc) < 2:
        return float("nan")
    return float(np.mean(np.diff(t[zc])))


def _log_decrement_damping(sig: np.ndarray) -> float:
    if len(sig) < 6:
        return float("nan")
    s = sig - np.mean(sig)
    peaks: List[float] = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > 0:
            peaks.append(float(s[i]))
    if len(peaks) < 2:
        return float("nan")
    deltas = [np.log(peaks[i] / peaks[i + 1])
              for i in range(len(peaks) - 1) if peaks[i + 1] > 1e-9]
    if not deltas:
        return float("nan")
    delta = float(np.mean(deltas))
    return float(delta / np.sqrt(4.0 * np.pi ** 2 + delta ** 2))


def case_metrics(case: TestCase, log: np.ndarray) -> dict:
    if len(log) == 0:
        return {"n_steps": 0}
    t, h, V, theta, phi, psi, p, q, r = (log[:, i] for i in range(9))
    out: dict = {
        "n_steps": int(len(log)),
        "h": _stats(h), "V": _stats(V),
        "theta": _stats(theta), "phi": _stats(phi), "psi": _stats(psi),
        "p": _stats(p), "q": _stats(q), "r": _stats(r),
    }
    if case.name == "trim_hold":
        out["drift_dh"] = float(np.max(np.abs(h - h[0])))
        out["drift_dV"] = float(np.max(np.abs(V - V[0])))
        out["drift_dtheta_deg"] = float(np.rad2deg(np.max(np.abs(theta - theta[0]))))
    elif case.name.startswith("elev_doublet"):
        out["q_peak"] = _peak(t, q)
        out["theta_peak"] = _peak(t, theta)
        out["period_h_s"] = _zero_crossing_period(t, h)
        out["damping_h"]  = _log_decrement_damping(h)
    elif case.name == "ail_doublet":
        out["p_peak"]   = _peak(t, p)
        out["phi_peak"] = _peak(t, phi)
    elif case.name == "rud_doublet":
        out["r_peak"]   = _peak(t, r)
        out["period_r_s"] = _zero_crossing_period(t, r)
        out["damping_r"]  = _log_decrement_damping(r)
    elif case.name == "ail_step_small":
        out["mean_phi_deg"]   = float(np.rad2deg(np.mean(phi)))
        out["mean_p_dps"]     = float(np.rad2deg(np.mean(p)))
        out["final_phi_deg"]  = float(np.rad2deg(phi[-1]))
        out["dh"] = float(h[-1] - h[0])
    elif case.name == "throttle_step":
        out["dV"] = float(V[-1] - V[0])
        out["dh"] = float(h[-1] - h[0])
    elif case.name == "elev_chirp":
        from numpy.fft import rfft, rfftfreq
        dt_loc = t[1] - t[0]
        win = np.hanning(len(t))
        ele = log[:, 10] - np.mean(log[:, 10])
        Q = rfft(q * win); E = rfft(ele * win)
        freqs = rfftfreq(len(t), dt_loc)
        for f0 in (0.5, 1.0, 1.5):
            idx = int(np.argmin(np.abs(freqs - f0)))
            out[f"gain_q_per_ele_at_{f0:.1f}Hz"] = float(abs(Q[idx]) / (abs(E[idx]) + 1e-9))
    return out


def divergence_pair(jsb: np.ndarray, jax_: np.ndarray) -> dict:
    n = min(len(jsb), len(jax_))
    if n == 0:
        return {k: float("nan") for k in
                ("rms_dh","rms_dV","rms_dtheta","rms_dphi","rms_dp","rms_dq","rms_dr")}
    a, b = jsb[:n], jax_[:n]
    return {
        "n_steps":    int(n),
        "rms_dh":     float(np.sqrt(np.mean((a[:, 1] - b[:, 1]) ** 2))),
        "rms_dV":     float(np.sqrt(np.mean((a[:, 2] - b[:, 2]) ** 2))),
        "rms_dtheta": float(np.sqrt(np.mean((a[:, 3] - b[:, 3]) ** 2))),
        "rms_dphi":   float(np.sqrt(np.mean((a[:, 4] - b[:, 4]) ** 2))),
        "rms_dp":     float(np.sqrt(np.mean((a[:, 6] - b[:, 6]) ** 2))),
        "rms_dq":     float(np.sqrt(np.mean((a[:, 7] - b[:, 7]) ** 2))),
        "rms_dr":     float(np.sqrt(np.mean((a[:, 8] - b[:, 8]) ** 2))),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

COL_NAMES = ["t_s", "h_m", "V_mps", "theta_rad", "phi_rad", "psi_rad",
             "p_radps", "q_radps", "r_radps",
             "ail_norm", "ele_norm", "rud_norm", "thr_norm"]


def save_csv(path: str, log: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COL_NAMES)
        w.writerows(log.tolist())


def plot_case(case: TestCase, jsb: np.ndarray, jax_: np.ndarray, out_path: str):
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    panels = [
        ("Altitude [m]",          1, 1.0),
        ("Airspeed [m/s]",        2, 1.0),
        ("Pitch theta [deg]",     3, 180.0 / np.pi),
        ("Roll phi [deg]",        4, 180.0 / np.pi),
        ("Yaw psi [deg]",         5, 180.0 / np.pi),
        ("Pitch rate q [deg/s]",  7, 180.0 / np.pi),
        ("Roll rate p [deg/s]",   6, 180.0 / np.pi),
        ("Yaw rate r [deg/s]",    8, 180.0 / np.pi),
    ]
    for ax, (label, col, scale) in zip(axes.flat, panels):
        if len(jsb):
            ax.plot(jsb[:, 0], jsb[:, col] * scale, label="JSBSim", lw=1.2)
        if len(jax_):
            ax.plot(jax_[:, 0], jax_[:, col] * scale, label="jax_sim", lw=1.2, ls="--")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time [s]"); axes[-1, 1].set_xlabel("time [s]")
    fig.suptitle(f"{case.name} — {case.description}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.97))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aircraft", default="px4_sih_uav")
    p.add_argument("--fdm-root", default="jsbsim_data")
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--alt-ft", type=float, default=328.084)
    p.add_argument("--speed-kts", type=float, default=38.88)
    p.add_argument("--output-dir", default="aero_baseline")
    p.add_argument("--cases", default="",
                   help="Comma-separated subset of case names; default = all.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trim: Trim = (0.0, 0.0758, 0.0, 0.3792)
    alpha_deg = -0.843
    fdm_root = args.fdm_root or None
    speed_m_s = args.speed_kts * 0.514444
    alt_m = args.alt_ft * M_PER_FT

    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in SUITE if c.name in wanted]
    else:
        cases = SUITE

    print(f"=== Aero validation suite ({len(cases)} cases @ {1/args.dt:.0f} Hz) ===")
    print(f"  trim = ail={trim[0]:+.4f} ele={trim[1]:+.4f} "
          f"rud={trim[2]:+.4f} thr={trim[3]:.4f}  alpha={alpha_deg:+.3f}deg")
    print(f"  output_dir = {args.output_dir}")
    print()

    all_metrics = {
        "config": {
            "aircraft": args.aircraft,
            "dt": args.dt,
            "alt_m": alt_m, "speed_m_s": speed_m_s,
            "trim_ail": trim[0], "trim_ele": trim[1],
            "trim_rud": trim[2], "trim_thr": trim[3],
            "alpha_deg": alpha_deg,
            "ground_guard_m": GROUND_GUARD_M,
        },
        "cases": {},
    }
    summary_rows: List[str] = []

    for case in cases:
        case_dir = os.path.join(args.output_dir, case.name)
        os.makedirs(case_dir, exist_ok=True)

        t0 = time.perf_counter()
        jsb_log, jsb_crashed = run_jsbsim(args.aircraft, fdm_root, case, args.dt,
                                          args.alt_ft, args.speed_kts, trim, alpha_deg)
        t_jsb = time.perf_counter() - t0

        t0 = time.perf_counter()
        jax_log, jax_crashed = run_jax_sim(case, args.dt, speed_m_s, alt_m, trim, alpha_deg)
        t_jax = time.perf_counter() - t0

        save_csv(os.path.join(case_dir, "jsbsim.csv"), jsb_log)
        save_csv(os.path.join(case_dir, "jax_sim.csv"), jax_log)
        plot_case(case, jsb_log, jax_log, os.path.join(case_dir, "compare.png"))

        m_jsb = case_metrics(case, jsb_log)
        m_jax = case_metrics(case, jax_log)
        div = divergence_pair(jsb_log, jax_log)
        all_metrics["cases"][case.name] = {
            "description": case.description,
            "duration_s":  case.duration,
            "jsbsim":  {"crashed": jsb_crashed, "wall_s": t_jsb, "n_rows": len(jsb_log), **m_jsb},
            "jax_sim": {"crashed": jax_crashed, "wall_s": t_jax, "n_rows": len(jax_log), **m_jax},
            "divergence": div,
        }

        marker = ""
        if jsb_crashed: marker += " [JSB-crashed]"
        if jax_crashed: marker += " [jax-crashed]"
        line = (f"  {case.name:<18}  dur={case.duration:5.1f}s  "
                f"dh={div['rms_dh']:6.2f}m  dV={div['rms_dV']:5.2f}m/s  "
                f"dtheta={np.rad2deg(div['rms_dtheta']):6.2f}deg  "
                f"dphi={np.rad2deg(div['rms_dphi']):6.2f}deg  "
                f"dq={np.rad2deg(div['rms_dq']):6.2f}d/s"
                f"{marker}")
        print(line)
        summary_rows.append(line)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  wrote {metrics_path}")

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Aero validation suite — independent maneuver tests\n")
        f.write(f"trim = {trim}, alpha={alpha_deg:.3f}deg, dt={args.dt}\n\n")
        for r in summary_rows:
            f.write(r + "\n")


if __name__ == "__main__":
    main()

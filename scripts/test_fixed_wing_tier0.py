#!/usr/bin/env python3
"""Validation suite for fixed-wing Tier 0 (spec §18.2).

Checks:
  a) Trim sanity: residuals ≈ 0 at the solved (alpha, ele, throttle).
  b) Linearization sanity: A/B shapes, |F_trim| / |M_trim| tiny.
  c) Doublet match: small-amplitude elevator doublet, Tier 0 vs Tier 1
     trajectories agree within the spec bar (pos < 0.5 m, attitude < 0.05 rad
     over 2 s).
  d) Gradient smoke: jax.grad over a Tier-0 rollout is finite.

Run:
    uv run python scripts/test_fixed_wing_tier0.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_sim.vehicles.fixed_wing import tier0, tier1
from jax_sim.vehicles.fixed_wing._shared import solve_trim, extract_tier0_coeffs
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_FIXED_WING
from jax_sim.utils.quaternion import quat_to_euler_jax


def banner(s: str) -> None:
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    # -------------------------------------------------------------------
    # (a) Trim sanity
    # -------------------------------------------------------------------
    banner("(a) Trim sanity")

    params = DEFAULT_FIXED_WING
    trim_state, trim_act = solve_trim(params, airspeed=20.0, altitude=100.0)
    flap_max = float(params.actuators.flap_max)

    # Unpack trim
    quat = trim_state[6:10]
    alpha = 2.0 * jnp.arctan2(quat[2], quat[0])  # pitch-only quat
    ele_norm = float(trim_act[1])
    ele_deg = np.rad2deg(ele_norm * flap_max)
    thr = float(trim_act[3])

    print(f"  alpha     = {float(np.rad2deg(alpha)):+.4f} deg")
    print(f"  elevator  = {ele_norm:+.6f}  (normalized)  /  {ele_deg:+.4f} deg")
    print(f"  throttle  = {thr:.6f}")

    # Residuals: re-evaluate forces_moments at trim and add body gravity.
    F, M = tier1.forces_moments(trim_state, params, jnp.zeros(3))
    mass = float(params.mass_props.mass)
    g = float(params.environment.gravity)
    Wx = -mass * g * float(jnp.sin(alpha))
    Wz = mass * g * float(jnp.cos(alpha))
    res = np.array([float(F[0]) + Wx, float(F[2]) + Wz, float(M[1])])
    print(f"  residuals = {res}")
    print(f"  |residual_max| = {np.max(np.abs(res)):.3e}")
    assert np.max(np.abs(res)) < 1e-3, "Trim did not converge tightly enough."

    # -------------------------------------------------------------------
    # (b) Linearization sanity
    # -------------------------------------------------------------------
    banner("(b) Linearization sanity")

    coeffs = extract_tier0_coeffs(params, trim_state, trim_act)
    print(f"  A shape = {tuple(coeffs.A.shape)} (expected (6, 17))")
    print(f"  B shape = {tuple(coeffs.B.shape)} (expected (6, 4))")
    print(f"  F_trim  = {np.asarray(coeffs.F_trim)}")
    print(f"  M_trim  = {np.asarray(coeffs.M_trim)}")
    # The aero+thrust wrench at trim balances gravity; |F_trim| ~= |mg|
    # in body z, so the absolute magnitude isn't tiny. What we check is the
    # *moment* near zero and that F[0] matches the body-x gravity component.
    print(f"  |F_trim|  = {float(jnp.linalg.norm(coeffs.F_trim)):.4f}")
    print(f"  |M_trim|  = {float(jnp.linalg.norm(coeffs.M_trim)):.3e}")
    assert coeffs.A.shape == (6, 17)
    assert coeffs.B.shape == (6, 4)
    assert float(jnp.linalg.norm(coeffs.M_trim)) < 1e-3, "M_trim not near zero."
    # F_trim balances gravity in body frame: F + W_body = 0
    body_grav = np.array([Wx, 0.0, Wz])
    F_plus_g = np.asarray(coeffs.F_trim) + body_grav
    print(f"  F_trim + W_body = {F_plus_g}  (expected ~0)")
    assert np.max(np.abs(F_plus_g)) < 1e-3, "Net wrench at trim not zero."

    # -------------------------------------------------------------------
    # (c) Doublet match (Tier 0 vs Tier 1)
    # -------------------------------------------------------------------
    banner("(c) Small-amplitude elevator doublet: Tier 0 vs Tier 1")

    dt = 0.004
    T = 2.0
    n = int(T / dt)
    amp = 0.02  # small enough to stay near trim

    def doublet_cmd(k):
        t = k * dt
        # +amp on [0, 0.5), -amp on [0.5, 1.0), 0 on [1.0, 2.0]
        ele = jnp.where(t < 0.5, amp, jnp.where(t < 1.0, -amp, 0.0))
        return jnp.array([trim_act[0], trim_act[1] + ele, trim_act[2], trim_act[3]])

    cmds = jnp.stack([doublet_cmd(k) for k in range(n)])  # (n, 4)

    t0_params = tier0.create_default_tier0(airspeed=20.0, altitude=100.0)

    def rollout_tier1(s0, cmds):
        def body(s, u):
            s2 = tier1.step(s, u, dt, params, jnp.zeros(3))
            return s2, s2
        _, traj = jax.lax.scan(body, s0, cmds)
        return traj

    def rollout_tier0(s0, cmds):
        def body(s, u):
            s2 = tier0.step(s, u, dt, t0_params, jnp.zeros(3))
            return s2, s2
        _, traj = jax.lax.scan(body, s0, cmds)
        return traj

    rollout_tier1_jit = jax.jit(rollout_tier1)
    rollout_tier0_jit = jax.jit(rollout_tier0)

    traj1 = rollout_tier1_jit(trim_state, cmds)
    traj0 = rollout_tier0_jit(trim_state, cmds)
    traj1.block_until_ready()
    traj0.block_until_ready()

    pos1 = np.asarray(traj1[:, 0:3])
    pos0 = np.asarray(traj0[:, 0:3])
    quat1 = np.asarray(traj1[:, 6:10])
    quat0 = np.asarray(traj0[:, 6:10])

    eul1 = np.asarray(jax.vmap(quat_to_euler_jax)(jnp.asarray(quat1)))
    eul0 = np.asarray(jax.vmap(quat_to_euler_jax)(jnp.asarray(quat0)))

    pos_err = np.linalg.norm(pos1 - pos0, axis=1)
    eul_err = np.linalg.norm(eul1 - eul0, axis=1)
    max_pos = float(pos_err.max())
    max_eul = float(eul_err.max())
    print(f"  max |pos_tier1 - pos_tier0|   = {max_pos:.4f} m   (bar: 0.5 m)")
    print(f"  max |euler_tier1 - euler_tier0| = {max_eul:.4f} rad (bar: 0.05 rad)")
    # Per-axis breakdown
    pos_axis = np.max(np.abs(pos1 - pos0), axis=0)
    eul_axis = np.max(np.abs(eul1 - eul0), axis=0)
    print(f"  per-axis max pos err (x,y,z) = {pos_axis}")
    print(f"  per-axis max eul err (r,p,y) = {eul_axis}")
    assert max_pos < 0.5, f"Tier 0 position diverged ({max_pos:.4f} m > 0.5 m)."
    assert max_eul < 0.05, f"Tier 0 attitude diverged ({max_eul:.4f} rad > 0.05 rad)."

    # -------------------------------------------------------------------
    # (d) Gradient smoke
    # -------------------------------------------------------------------
    banner("(d) Gradient smoke test (Tier 0 BPTT)")

    def terminal_z(amp_):
        cmd_seq = jnp.stack([
            jnp.array([
                trim_act[0],
                trim_act[1] + jnp.where(k * dt < 0.5, amp_,
                                       jnp.where(k * dt < 1.0, -amp_, 0.0)),
                trim_act[2],
                trim_act[3],
            ])
            for k in range(n)
        ])
        traj = rollout_tier0(trim_state, cmd_seq)
        return traj[-1, 2]  # terminal z (NED)

    grad = float(jax.grad(terminal_z)(0.02))
    print(f"  d(terminal_z)/d(amp) at amp=0.02 = {grad:+.6f}")
    assert np.isfinite(grad), "Gradient is non-finite."

    banner("ALL ASSERTIONS PASSED")
    print(f"  Trim: alpha={float(np.rad2deg(alpha)):+.3f} deg, ele_norm={ele_norm:+.5f}, thr={thr:.5f}")
    print(f"  Doublet match: pos {max_pos:.4f} m, attitude {max_eul:.4f} rad")


if __name__ == "__main__":
    main()

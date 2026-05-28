#!/usr/bin/env python3
"""Extract dimensionless stability + control derivatives from jax_sim's aero,
linearized around a level cruise trim, so they can be transcribed into a
JSBSim aircraft XML.

Process:
  1. Find trim controls (elevator, throttle) for steady level flight at
     V = trim_speed, alpha = trim_alpha (numerical search).
  2. Numerically differentiate compute_fixed_wing_aero w.r.t. alpha, beta,
     body rates p/q/r, and control deflections aileron/elevator/rudder.
  3. Print results in JSBSim convention: dimensionless coefficients
     using reference S, b, c-bar.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, fsolve

from jax_sim.vehicles.fixed_wing._aero_segment import compute_fixed_wing_aero
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_AIRCRAFT
from jax_sim.physics.constants import G as _G, RHO as _RHO
from jax_sim.vehicles.fixed_wing import presets as C

V_TRIM = 20.0   # m/s, jax_sim's design cruise
ALT = 100.0     # m


# Reference geometry (must match JSBSim XML)
S = float(C.WING_AREA)        # 0.18 m^2
B = float(C.WING_SPAN)        # 0.86 m
CBAR = float(C.CHORD)         # 0.21 m
RHO = float(_RHO)             # 1.225
MASS = float(C.MASS)          # 1.0 kg
G = float(_G)                 # 9.81


def body_aero(alpha, beta, p, q, r, ail, ele, rud, thr, V=V_TRIM):
    """Return (F_body, M_body) from jax_sim aero at the given state."""
    u = V * jnp.cos(alpha) * jnp.cos(beta)
    v = V * jnp.sin(beta)
    w = V * jnp.sin(alpha) * jnp.cos(beta)
    v_body = jnp.array([u, v, w])
    omega = jnp.array([p, q, r])
    return compute_fixed_wing_aero(
        v_body, omega, ail, ele, rud, thr, ALT, DEFAULT_AIRCRAFT
    )


def trim_residual(alpha, ele, thr):
    """Residuals of (Fx + Thrust - Drag = 0, Fz + W = 0, My = 0) at level trim."""
    F, M = body_aero(alpha, 0.0, 0.0, 0.0, 0.0, 0.0, ele, 0.0, thr)
    # Body x-force balanced by thrust (already inside compute_fixed_wing_aero? no:
    # compute_fixed_wing_aero returns aero only; thrust comes from dynamics.py).
    # So Fx_total = F[0] + thrust*T_MAX -- but throttle is already passed for
    # slipstream effect. We need to add thrust explicitly here.
    T = thr * C.T_MAX
    # Body-frame gravity (level attitude, theta = alpha for body-axis trim)
    # For wings-level alpha trim, gravity in body frame:
    Wx = -MASS * G * jnp.sin(alpha)
    Wz = MASS * G * jnp.cos(alpha)
    res_x = F[0] + T + Wx
    res_z = F[2] + Wz
    res_m = M[1]
    return jnp.array([res_x, res_z, res_m])


def find_trim():
    """Solve for (alpha, elevator, throttle) that produces level steady flight."""
    def f(x):
        a, e, t = x
        return np.asarray(trim_residual(a, e, t))
    x0 = np.array([np.deg2rad(2.0), 0.0, 0.3])
    sol, info, ier, msg = fsolve(f, x0, full_output=True, xtol=1e-9)
    if ier != 1:
        print(f"[trim] fsolve warning: {msg}")
    a, e, t = sol
    res = f(sol)
    print(f"[trim] alpha = {np.rad2deg(a):+.3f} deg")
    print(f"[trim] elevator (norm) = {e:+.4f}")
    print(f"[trim] throttle = {t:.4f}")
    print(f"[trim] residual = {res}")
    return float(a), float(e), float(t)


def deriv(fn, x0, eps):
    """Central difference of fn(x) at x0 (returns vector)."""
    return (np.asarray(fn(x0 + eps)) - np.asarray(fn(x0 - eps))) / (2.0 * eps)


def main():
    print(f"Reference: S = {S} m^2, b = {B} m, c = {CBAR} m, V_trim = {V_TRIM} m/s")
    a_t, e_t, t_t = find_trim()
    q_dyn = 0.5 * RHO * V_TRIM * V_TRIM
    qS = q_dyn * S

    def F_M_at(a, b, p, q, r, ail, ele, rud, thr):
        F, M = body_aero(a, b, p, q, r, ail, ele, rud, thr)
        return np.concatenate([np.asarray(F), np.asarray(M)])

    # Baseline at trim
    base = F_M_at(a_t, 0.0, 0.0, 0.0, 0.0, 0.0, e_t, 0.0, t_t)
    Fx0, Fy0, Fz0, Mx0, My0, Mz0 = base
    # Convert body force to wind-axis lift/drag at trim alpha:
    ca, sa = np.cos(a_t), np.sin(a_t)
    L0 = -Fz0 * ca + Fx0 * sa
    D0 = -Fx0 * ca - Fz0 * sa
    CL0 = L0 / qS
    CD0 = D0 / qS
    Cm0 = My0 / (qS * CBAR)
    print(f"\nTrim coefficients: CL0 = {CL0:+.4f}, CD0 = {CD0:+.4f}, Cm0 = {Cm0:+.4f}")

    def show(name, dF, ref):
        """Print 6-component derivative scaled by qS and ref length."""
        # dF = [dFx, dFy, dFz, dMx, dMy, dMz]
        # Wind-axis L = -Fz*ca + Fx*sa (use trim alpha for transform)
        dL = -dF[2] * ca + dF[0] * sa
        dD = -dF[0] * ca - dF[2] * sa
        dY = dF[1]
        dl = dF[3]
        dm = dF[4]
        dn = dF[5]
        CL = dL / qS / ref
        CD = dD / qS / ref
        CY = dY / qS / ref
        Cl = dl / (qS * B) / ref
        Cm = dm / (qS * CBAR) / ref
        Cn = dn / (qS * B) / ref
        print(f"{name}: CL={CL:+.4f} CD={CD:+.4f} CY={CY:+.4f} "
              f"Cl={Cl:+.4f} Cm={Cm:+.4f} Cn={Cn:+.4f}")

    # Alpha derivative
    da = np.deg2rad(0.5)
    d = deriv(lambda x: F_M_at(a_t + x, 0, 0, 0, 0, 0, e_t, 0, t_t), 0.0, da)
    show("d/d_alpha  [per rad]", d, 1.0)

    # Beta
    db = np.deg2rad(0.5)
    d = deriv(lambda x: F_M_at(a_t, x, 0, 0, 0, 0, e_t, 0, t_t), 0.0, db)
    show("d/d_beta   [per rad]", d, 1.0)

    # Body rates p, q, r (non-dimensionalized by b/2V or c/2V)
    pscale = B / (2.0 * V_TRIM)
    qscale = CBAR / (2.0 * V_TRIM)
    rscale = B / (2.0 * V_TRIM)
    dp = 0.5
    d = deriv(lambda x: F_M_at(a_t, 0, x, 0, 0, 0, e_t, 0, t_t), 0.0, dp)
    show(f"d/d_p_hat  [per rad, p_hat=pb/2V; scaling={pscale:.4f}]", d / pscale, 1.0)
    d = deriv(lambda x: F_M_at(a_t, 0, 0, x, 0, 0, e_t, 0, t_t), 0.0, dp)
    show(f"d/d_q_hat  [per rad, q_hat=qc/2V; scaling={qscale:.4f}]", d / qscale, 1.0)
    d = deriv(lambda x: F_M_at(a_t, 0, 0, 0, x, 0, e_t, 0, t_t), 0.0, dp)
    show(f"d/d_r_hat  [per rad, r_hat=rb/2V; scaling={rscale:.4f}]", d / rscale, 1.0)

    # Controls (in normalized [-1,1] -- so derivatives are per unit norm)
    duc = 0.1
    d = deriv(lambda x: F_M_at(a_t, 0, 0, 0, 0, x, e_t, 0, t_t), 0.0, duc)
    show("d/d_aileron[per norm]", d, 1.0)
    d = deriv(lambda x: F_M_at(a_t, 0, 0, 0, 0, 0, e_t + x, 0, t_t), 0.0, duc)
    show("d/d_elev   [per norm]", d, 1.0)
    d = deriv(lambda x: F_M_at(a_t, 0, 0, 0, 0, 0, e_t, x, t_t), 0.0, duc)
    show("d/d_rudder [per norm]", d, 1.0)

    # Throttle (thrust effect on aero via slipstream)
    d = deriv(lambda x: F_M_at(a_t, 0, 0, 0, 0, 0, e_t, 0, t_t + x), 0.0, 0.05)
    show("d/d_throttle[per unit throttle, aero-only]", d, 1.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose the +60% pitch / +43% yaw / -20% throttle divergence between
jax_sim and JSBSim by directly probing aero moments at quasi-static
deflections and comparing against the JSBSim XML linear prediction.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing._aero_segment import compute_fixed_wing_aero
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_AIRCRAFT
from jax_sim.physics.constants import RHO as _RHO
from jax_sim.vehicles.fixed_wing import presets as C

V_TRIM = 20.0
ALT = 100.0
S = float(C.WING_AREA)
B = float(C.WING_SPAN)
CBAR = float(C.CHORD)
RHO = float(_RHO)
QBAR = 0.5 * RHO * V_TRIM ** 2

# JSBSim XML values (from px4_sih_uav.xml)
CMDE_XML = -0.4442
CNDR_XML =  0.1232
CLDA_XML =  0.1065

ALPHA_TRIM = np.deg2rad(-0.843)
ELE_TRIM   =  0.0758
THR_TRIM   =  0.3792


def body_aero(alpha, ele, rud, ail, thr, p=0.0, q=0.0, r=0.0):
    u = V_TRIM * np.cos(alpha)
    w = V_TRIM * np.sin(alpha)
    v_body = jnp.array([u, 0.0, w])
    omega = jnp.array([p, q, r])
    F, M = compute_fixed_wing_aero(v_body, omega, ail, ele, rud, thr,
                                    ALT, DEFAULT_AIRCRAFT)
    return np.asarray(F), np.asarray(M)


def cm_pitch(alpha, ele, thr):
    _, M = body_aero(alpha, ele, 0.0, 0.0, thr)
    return float(M[1]) / (QBAR * S * CBAR)


def cn_yaw(alpha, rud, thr):
    _, M = body_aero(alpha, ELE_TRIM, rud, 0.0, thr)
    return float(M[2]) / (QBAR * S * B)


def cl_roll(alpha, ail, thr):
    _, M = body_aero(alpha, ELE_TRIM, 0.0, ail, thr)
    return float(M[0]) / (QBAR * S * B)


def cx_force(alpha, ele, thr):
    F, _ = body_aero(alpha, ele, 0.0, 0.0, thr)
    return float(F[0]) / (QBAR * S)


print(f"Trim: V={V_TRIM} m/s, alpha={np.rad2deg(ALPHA_TRIM):.2f} deg, "
      f"ele={ELE_TRIM:.4f}, thr={THR_TRIM:.4f}")
print(f"qbar = {QBAR:.2f} Pa, S = {S:.4f} m^2, c = {CBAR:.4f} m, b = {B:.4f} m")
print()

# ==========================================================================
# Elevator: jax_sim vs XML linear prediction (Cmde * de)
# ==========================================================================
print("=" * 78)
print("PITCH: jax_sim Cm vs JSBSim linear prediction (Cm = Cmde * (ele - ele_trim))")
print("=" * 78)
cm_trim = cm_pitch(ALPHA_TRIM, ELE_TRIM, THR_TRIM)
print(f"  Cm at trim = {cm_trim:+.6f}  (should be ~0)")
print()
print(f"  {'delta_ele':>12}  {'Cm_jax':>10}  {'Cm_xml':>10}  {'ratio jax/xml':>14}")
for d_ele in [-0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15]:
    cm_jax = cm_pitch(ALPHA_TRIM, ELE_TRIM + d_ele, THR_TRIM) - cm_trim
    cm_xml = CMDE_XML * d_ele
    ratio = cm_jax / cm_xml if abs(cm_xml) > 1e-9 else float("nan")
    print(f"  {d_ele:>+12.3f}  {cm_jax:>+10.5f}  {cm_xml:>+10.5f}  {ratio:>14.3f}")

# Effective Cmde from jax_sim at small deflection
d_ele = 0.01
cm_d = cm_pitch(ALPHA_TRIM, ELE_TRIM + d_ele, THR_TRIM) - cm_pitch(ALPHA_TRIM, ELE_TRIM - d_ele, THR_TRIM)
cmde_jax = cm_d / (2 * d_ele)
print(f"\n  Cmde (jax_sim, central diff at +/-{d_ele}): {cmde_jax:+.4f}")
print(f"  Cmde (JSBSim XML):                         {CMDE_XML:+.4f}")

# ==========================================================================
# RUDDER
# ==========================================================================
print()
print("=" * 78)
print("YAW: jax_sim Cn vs JSBSim linear (Cn = Cndr * d_rud)")
print("=" * 78)
cn_trim = cn_yaw(ALPHA_TRIM, 0.0, THR_TRIM)
print(f"  Cn at trim (rud=0) = {cn_trim:+.6f}")
print()
print(f"  {'delta_rud':>12}  {'Cn_jax':>10}  {'Cn_xml':>10}  {'ratio jax/xml':>14}")
for d_rud in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
    cn_jax = cn_yaw(ALPHA_TRIM, d_rud, THR_TRIM) - cn_trim
    cn_xml = CNDR_XML * d_rud
    ratio = cn_jax / cn_xml if abs(cn_xml) > 1e-9 else float("nan")
    print(f"  {d_rud:>+12.3f}  {cn_jax:>+10.5f}  {cn_xml:>+10.5f}  {ratio:>14.3f}")

# ==========================================================================
# AILERON  (we expect this to MATCH because the dynamics matched)
# ==========================================================================
print()
print("=" * 78)
print("ROLL: jax_sim Cl vs JSBSim linear (Cl = Clda * d_ail) [reference: matches]")
print("=" * 78)
cl_trim = cl_roll(ALPHA_TRIM, 0.0, THR_TRIM)
print(f"  Cl at trim (ail=0) = {cl_trim:+.6f}")
print()
print(f"  {'delta_ail':>12}  {'Cl_jax':>10}  {'Cl_xml':>10}  {'ratio jax/xml':>14}")
for d_ail in [-0.30, -0.15, -0.05, 0.05, 0.15, 0.30]:
    cl_jax = cl_roll(ALPHA_TRIM, d_ail, THR_TRIM) - cl_trim
    cl_xml = CLDA_XML * d_ail
    ratio = cl_jax / cl_xml if abs(cl_xml) > 1e-9 else float("nan")
    print(f"  {d_ail:>+12.3f}  {cl_jax:>+10.5f}  {cl_xml:>+10.5f}  {ratio:>14.3f}")

# ==========================================================================
# THROTTLE  (jax climbs 20% less under +0.2 throttle)
# ==========================================================================
print()
print("=" * 78)
print("THROTTLE: jax_sim thrust vs JSBSim (linear: thrust = thr * T_max)")
print("=" * 78)
print(f"  T_max = {C.T_MAX} N, lever from CG = 0 (thrust along body-x)")
print()
print(f"  {'thr':>8}  {'jax_Fx (N)':>12}  {'JSB_Fx (N)':>12}  {'jax_Cm':>10}  {'note':<20}")
for thr in [0.0, 0.1, 0.2, 0.379, 0.5, 0.6]:
    Fx, M = body_aero(ALPHA_TRIM, ELE_TRIM, 0.0, 0.0, thr)
    # JSBSim applies thrust as external reaction: Fx_thrust = thr * T_max
    # jax_sim applies it inside dynamics.py: thrust = thr * T_max
    # The aero force here EXCLUDES thrust; what about slipstream effect on aero?
    cm_thr = float(M[1]) / (QBAR * S * CBAR)
    fx_thrust = thr * C.T_MAX
    print(f"  {thr:>8.3f}  {float(Fx)+fx_thrust:>12.4f}  {fx_thrust:>12.4f}  {cm_thr:>+10.5f}  {'(jax includes aero+slip)' if thr > 0 else ''}")

# Lift change vs throttle (slipstream lift on tail / segments)
print()
print(f"  {'thr':>8}  {'Fz_aero (N)':>12}  {'CL eff':>10}")
for thr in [0.0, 0.1, 0.2, 0.379, 0.5, 0.6]:
    F, _ = body_aero(ALPHA_TRIM, ELE_TRIM, 0.0, 0.0, thr)
    ca, sa = np.cos(ALPHA_TRIM), np.sin(ALPHA_TRIM)
    L = -float(F[2]) * ca + float(F[0]) * sa
    CL = L / (QBAR * S)
    print(f"  {thr:>8.3f}  {float(F[2]):>+12.4f}  {CL:>+10.4f}")

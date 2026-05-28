"""Aerodynamic model for fixed-wing aircraft.

This module provides aerodynamic force and moment calculations.

The main entry point is `compute_fixed_wing_aero()` which computes all
aerodynamic forces and moments for a fixed-wing aircraft using the
PX4 SIH-style segment model.

For backwards compatibility, `compute_aero_segment()` is still available
but deprecated.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing.params import FixedWingParams
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_FIXED_WING
from jax_sim.physics.constants import RHO
from jax_sim.vehicles.fixed_wing._aero_segment_lib import compute_segment_forces

AircraftParams = FixedWingParams
DEFAULT_AIRCRAFT = DEFAULT_FIXED_WING


@jax.jit
def compute_fixed_wing_aero(
    v_body: jnp.ndarray,
    omega: jnp.ndarray,
    aileron: float,
    elevator: float,
    rudder: float,
    throttle: float,
    altitude: float = 0.0,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute total aerodynamic forces and moments for fixed-wing aircraft.

    Uses the PX4 SIH-style aerodynamic segment model with:
    - Wing segments with -4 deg incidence and dihedral
    - Tailplane with propeller slipstream
    - Vertical fin with propeller slipstream
    - Fuselage drag

    Args:
        v_body: Velocity in body frame [u, v, w] [m/s]
        omega: Angular velocity in body frame [p, q, r] [rad/s]
        aileron: Aileron deflection [-1, 1] normalized
        elevator: Elevator deflection [-1, 1] normalized
        rudder: Rudder deflection [-1, 1] normalized
        throttle: Throttle [0, 1] normalized
        altitude: Altitude above sea level [m]
        aircraft: Aircraft configuration (segments, limits, propulsion)

    Returns:
        Tuple of:
            F_aero: Total aerodynamic force in body frame [Fx, Fy, Fz] [N]
            M_aero: Total aerodynamic moment in body frame [Mx, My, Mz] [Nm]
    """
    # Convert normalized commands to radians
    flap_max = aircraft.actuators.flap_max
    ail_rad = aileron * flap_max
    ele_rad = elevator * flap_max
    rud_rad = rudder * flap_max

    # Thrust for slipstream calculation
    thrust = throttle * aircraft.propulsion.t_max

    # Compute forces from each segment
    # Note: aileron is positive for right roll (left aileron down, right up)
    # Note: elevator sign is already handled by cascade_pid.py (line 111)
    #       which negates the rate controller output for correct pitch response
    segments = aircraft.segments
    F_wing_l, M_wing_l = compute_segment_forces(
        segments.wing_left, v_body, omega, altitude, ail_rad, 0.0
    )
    F_wing_r, M_wing_r = compute_segment_forces(
        segments.wing_right, v_body, omega, altitude, -ail_rad, 0.0
    )
    F_tail, M_tail = compute_segment_forces(
        segments.tailplane,
        v_body,
        omega,
        altitude,
        ele_rad,
        thrust,  # No negation - done by controller
    )
    F_fin, M_fin = compute_segment_forces(
        segments.fin, v_body, omega, altitude, rud_rad, thrust
    )
    F_fuse, M_fuse = compute_segment_forces(
        segments.fuselage, v_body, omega, altitude, 0.0, 0.0
    )

    # Sum all contributions
    F_aero = F_wing_l + F_wing_r + F_tail + F_fin + F_fuse
    M_aero = M_wing_l + M_wing_r + M_tail + M_fin + M_fuse

    return F_aero, M_aero


# =============================================================================
# DEPRECATED: Old simple aerodynamic model (kept for backwards compatibility)
# =============================================================================

@jax.jit
def compute_aero_segment(v_local_body, area, control_deflection):
    """Compute aerodynamic force for one segment.

    DEPRECATED: Use compute_fixed_wing_aero() instead for accurate physics.

    This simplified model does not include:
    - Zero-lift angle of attack
    - Pitching moment
    - Propeller slipstream
    - Proper stall modeling

    Args:
        v_local_body: Local airspeed at the segment [u, v, w]
        area: Segment area (m^2)
        control_deflection: Control surface deflection (radians)

    Returns:
        Force vector [Fx, Fy, Fz] in body frame
    """
    # 1. Airspeed and angle of attack (alpha)
    u, v, w = v_local_body
    V_sq = jnp.sum(v_local_body**2)
    V = jnp.sqrt(V_sq + 1e-6)  # Avoid divide-by-zero

    # Alpha: vertical-plane angle (atan2(w, u))
    # w is down velocity, u is forward velocity
    alpha = jnp.arctan2(w, u)

    # 2. Effective alpha (includes control deflection)
    # Down deflection acts like increased alpha
    alpha_eff = alpha + control_deflection

    # 3. Stall model (PX4-style tanh blending)
    # Low angles: Lift = C_L_alpha * alpha
    # High angles (stall): flat plate theory (sin(2a))

    # Blending factor: transitions near +/- 20 deg (0.35 rad)
    stall_angle = 0.35
    sigmoid = 0.5 * (1.0 + jnp.tanh((jnp.abs(alpha_eff) - stall_angle) * 10.0))

    # Linear region (linear lift) - using corrected CL_alpha for finite wing
    CL_alpha = 3.8  # Corrected for AR ~4 (was 5.0)
    CL_linear = CL_alpha * alpha_eff
    CD_linear = 0.04 + 0.09 * alpha_eff**2  # Updated coefficients

    # Stall region (flat plate)
    CL_flat = 1.0 * jnp.sin(2.0 * alpha_eff)  # Simple post-stall lift
    CD_flat = 1.98 * jnp.sin(alpha_eff) ** 2 + 0.04  # High post-stall drag

    # Blend
    CL = (1.0 - sigmoid) * CL_linear + sigmoid * CL_flat
    CD = (1.0 - sigmoid) * CD_linear + sigmoid * CD_flat

    # 4. Compute forces (wind axes -> body axes)
    q_dyn = 0.5 * RHO * V_sq * area

    Lift = q_dyn * CL
    Drag = q_dyn * CD

    # Lift is perpendicular to velocity, drag is parallel.
    # Rotate by alpha into body frame (Fx, Fz)
    # Fx = -Drag * cos(a) + Lift * sin(a)
    # Fz = -Drag * sin(a) - Lift * cos(a)

    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    Fx = -Drag * ca + Lift * sa
    Fz = -Drag * sa - Lift * ca

    # Lateral force (side force) - simple damping
    Fy = -0.5 * RHO * V * area * v * 0.5  # Simple lateral drag

    return jnp.array([Fx, Fy, Fz])

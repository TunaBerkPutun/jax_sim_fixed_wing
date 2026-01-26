"""Aerodynamic segment model based on PX4 SIH aero.hpp.

This module implements a physics-based aerodynamic model for wing, tailplane,
and fin segments. It captures:
- Full 360 degree angle of attack behavior
- Stall modeling with leading/trailing edge separation
- Control surface (flap) deflection effects
- Propeller slipstream effects
- Finite wing corrections

Reference:
    Khan, Waqas, supervised by Meyer Nahon "Dynamics modeling of agile fixed-wing
    unmanned aerial vehicles." McGill University, PhD thesis, 2016.
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


class AeroSegmentParams(NamedTuple):
    """Parameters defining an aerodynamic segment.

    Attributes:
        span: Segment span [m]
        mac: Mean aerodynamic chord [m]
        alpha_0: Zero-lift angle of attack [rad]
        position: Position of aerodynamic center from CG [x, y, z] in body frame [m]
        dihedral: Dihedral angle [rad] (0 for horizontal, -pi/2 for vertical fin)
        aspect_ratio: Aspect ratio of the wing/surface (use -1 to compute from span/mac)
        flap_chord: Control surface chord length [m] (0 if no control surface)
        prop_radius: Propeller radius for slipstream [m] (-1 if no slipstream)
        cl_alpha: 2D lift curve slope [1/rad] (default 2*pi for flat plate)
        alpha_max: Maximum angle of attack before stall [rad] (0 to use table)
        alpha_min: Minimum angle of attack before stall [rad] (0 to use table)
    """
    span: float
    mac: float
    alpha_0: float
    position: jnp.ndarray  # [x, y, z]
    dihedral: float
    aspect_ratio: float
    flap_chord: float
    prop_radius: float
    cl_alpha: float = 2.0 * jnp.pi
    alpha_max: float = 0.0
    alpha_min: float = 0.0


# Physical constants
RHO_SEA_LEVEL = 1.225  # Air density at sea level [kg/m^3]
P0 = 101325.0  # Pressure at sea level [Pa]
R_AIR = 287.04  # Gas constant for air [J/kg/K]
T0_K = 288.15  # Temperature at sea level [K]
TEMP_GRADIENT = -6.5e-3  # Temperature gradient [K/m]

# Aerodynamic constants
KV = jnp.pi  # Total vortex lift parameter
CD0 = 0.04  # Zero-lift drag coefficient
CD90 = 1.98  # 90 degree AOA drag coefficient
ALPHA_BLEND = jnp.pi / 18.0  # 10 degrees blending region
K0 = 0.87  # Oswald efficiency factor

# Flap effectiveness polynomial coefficients (1/rad)
ETA_POLY = jnp.array([0.0535, -0.2688, 0.5817])

# Semi-empirical coefficient tables for flat plates (function of aspect ratio)
AR_TABLE = jnp.array([0.1666, 0.333, 0.4, 0.5, 1.0, 1.25, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
AFS_TABLE = jnp.array([49.0, 54.0, 56.0, 48.0, 40.0, 29.0, 27.0, 25.0, 24.0, 22.0, 22.0, 20.0])  # Stall angle [deg]


def _interp_table(x: float, x_table: jnp.ndarray, y_table: jnp.ndarray) -> float:
    """Linear interpolation with clamping at table boundaries."""
    return jnp.interp(x, x_table, y_table)


def _compute_air_density(altitude: float) -> float:
    """Compute air density using ISA model.

    Args:
        altitude: Altitude above sea level [m]

    Returns:
        Air density [kg/m^3]
    """
    temperature = T0_K + TEMP_GRADIENT * altitude
    pressure = P0 * jnp.power(1.0 - 0.0065 * altitude / T0_K, 5.2561)
    return pressure / (R_AIR * temperature)


def _compute_rotation_matrix(dihedral: float) -> jnp.ndarray:
    """Compute rotation matrix from segment frame to body frame.

    Args:
        dihedral: Dihedral angle [rad]

    Returns:
        3x3 rotation matrix
    """
    c = jnp.cos(dihedral)
    s = jnp.sin(dihedral)
    # Rotation about X axis (roll)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def _compute_derived_params(params: AeroSegmentParams) -> Tuple[float, float, float, float, float]:
    """Compute derived aerodynamic parameters.

    Args:
        params: Segment parameters

    Returns:
        Tuple of (ar, kp, kn, alpha_max, alpha_min, kD)
    """
    # Aspect ratio
    ar = jnp.where(
        params.aspect_ratio <= 0,
        params.span / params.mac,
        params.aspect_ratio
    )

    # Lift curve slope corrected for finite wing
    kp = params.cl_alpha / (1.0 + 2.0 * (ar + 4.0) / (ar * (ar + 2.0)))

    # Normal force coefficient parameter
    kn = 0.41 * (1.0 - jnp.exp(-17.0 / ar))

    # Induced drag factor
    kD = 1.0 / (jnp.pi * K0 * ar)

    # Stall angles from table if not specified
    afs_deg = _interp_table(ar, AR_TABLE, AFS_TABLE)
    afs_rad = jnp.deg2rad(afs_deg)

    alpha_max = jnp.where(
        jnp.abs(params.alpha_max) < 1e-3,
        afs_rad,
        params.alpha_max
    )
    alpha_min = jnp.where(
        jnp.abs(params.alpha_min) < 1e-3,
        -afs_rad,
        params.alpha_min
    )

    return ar, kp, kn, alpha_max, alpha_min, kD


def _compute_flap_effectiveness(deflection: float, mac: float, flap_chord: float) -> Tuple[float, float]:
    """Compute control surface effectiveness.

    Args:
        deflection: Flap deflection angle [rad]
        mac: Mean aerodynamic chord [m]
        flap_chord: Flap chord length [m]

    Returns:
        Tuple of (eta_f, tau_f) - effectiveness and lift contribution factor
    """
    # Limit deflection
    def_a = jnp.minimum(jnp.abs(deflection), jnp.deg2rad(70.0))

    # Second order fit for effectiveness
    eta_f = def_a * def_a * ETA_POLY[0] + def_a * ETA_POLY[1] + ETA_POLY[2]

    # Flap geometry factor
    cf_ratio = jnp.clip(flap_chord / mac, 0.0, 0.999)
    theta_f = jnp.arccos(2.0 * cf_ratio - 1.0)
    tau_f = 1.0 - (theta_f - jnp.sin(theta_f)) / jnp.pi

    return eta_f, tau_f


def _compute_low_aoa_coefficients(
    alpha_eff: float,
    kp: float,
    kn: float,
) -> Tuple[float, float, float]:
    """Compute lift, drag, and moment coefficients for low angle of attack.

    Uses the model from Khan's thesis for attached flow.

    Args:
        alpha_eff: Effective angle of attack [rad]
        kp: Lift curve slope (finite wing corrected)
        kn: Normal force parameter

    Returns:
        Tuple of (CL, CD, CM)
    """
    sa = jnp.sin(alpha_eff)
    ca = jnp.cos(alpha_eff)

    # Lift coefficient (potential + vortex)
    CL = kp * sa * ca * ca + KV * jnp.abs(sa) * sa * ca

    # Drag coefficient (parabolic polar)
    CD = CD0 + jnp.abs(CL * jnp.tan(alpha_eff))

    # Pitching moment coefficient (about quarter chord)
    CM = -0.0625 * kp * sa * ca + 0.17 * KV * jnp.abs(sa) * sa

    return CL, CD, CM


def _compute_high_aoa_coefficients(
    alpha: float,
    kn: float,
    deflection: float = 0.0,
    mac: float = 1.0,
    flap_chord: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute lift, drag, and moment coefficients for high angle of attack (stall).

    Uses flat plate theory for separated flow.

    Args:
        alpha: Angle of attack [rad]
        kn: Normal force parameter
        deflection: Flap deflection [rad]
        mac: Mean aerodynamic chord [m]
        flap_chord: Flap chord [m]

    Returns:
        Tuple of (CL, CD, CM)
    """
    # Effective chord with deflected flap
    mac_eff = jnp.sqrt(
        (mac - flap_chord) ** 2 + flap_chord ** 2 +
        2.0 * (mac - flap_chord) * flap_chord * jnp.cos(jnp.abs(deflection))
    )

    # Adjusted alpha for flap geometry
    alpha_adj = alpha + jnp.arcsin(flap_chord / mac_eff * jnp.sin(deflection))

    # CD90 adjusted for flap deflection
    cd90_eff = CD90 + 0.21 * deflection - 0.0426 * deflection * deflection

    # Normal coefficient (flat plate)
    sa = jnp.sin(alpha_adj)
    ca = jnp.cos(alpha_adj)
    CN = cd90_eff * sa * (1.0 / (0.56 + 0.44 * jnp.abs(sa)) - kn)

    # Tangential coefficient
    CT = 0.5 * CD0 * ca

    # Convert to lift/drag
    CL = CN * ca - CT * sa
    CD = CN * sa + CT * ca

    # Pitching moment
    CM = -CN * (0.25 - 7.0 / 40.0 * (1.0 - 2.0 / jnp.pi * jnp.abs(alpha_adj)))

    return CL, CD, CM


@jax.jit
def compute_segment_forces(
    params: AeroSegmentParams,
    v_body: jnp.ndarray,
    omega: jnp.ndarray,
    altitude: float,
    deflection: float,
    thrust: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute aerodynamic forces and moments for a segment.

    Args:
        params: Segment parameters
        v_body: Velocity in body frame [u, v, w] [m/s]
        omega: Angular velocity in body frame [p, q, r] [rad/s]
        altitude: Altitude above sea level [m]
        deflection: Control surface deflection [rad]
        thrust: Thrust force for slipstream calculation [N]

    Returns:
        Tuple of:
            F_body: Force in body frame [Fx, Fy, Fz] [N]
            M_body: Moment about CG in body frame [Mx, My, Mz] [Nm]
    """
    # Air density at altitude
    rho = _compute_air_density(altitude)

    # Rotation from segment to body frame
    C_BS = _compute_rotation_matrix(params.dihedral)

    # Local velocity at segment (includes rotational component)
    v_local = v_body + jnp.cross(omega, params.position)

    # Transform to segment frame
    v_S = C_BS.T @ v_local

    # Add propeller slipstream if applicable
    v_slip = jnp.where(
        params.prop_radius > 1e-4,
        jnp.sqrt(jnp.maximum(0.0, 2.0 * thrust / (rho * jnp.pi * params.prop_radius ** 2))),
        0.0
    )
    v_S = v_S.at[0].add(v_slip)

    # Velocity in XZ plane (for angle of attack)
    vxz2 = v_S[0] ** 2 + v_S[2] ** 2
    vxz = jnp.sqrt(vxz2 + 1e-8)

    # Handle zero velocity case
    def zero_velocity_case():
        return jnp.zeros(3), jnp.zeros(3)

    def normal_case():
        # Angle of attack (relative to segment)
        alpha = jnp.arctan2(v_S[2], v_S[0]) - params.alpha_0

        # Get derived parameters
        ar, kp, kn, alpha_max, alpha_min, kD = _compute_derived_params(params)

        # Effective alpha with control surface
        flap_ratio = jnp.clip(params.flap_chord / params.mac, 0.0, 0.999)

        # Full flap case vs partial flap
        alpha_eff = jnp.where(
            flap_ratio > 0.999,
            alpha + deflection,
            alpha + _compute_flap_delta_alpha(deflection, kp, params.mac, params.flap_chord)
        )

        # Compute coefficients for low and high AOA
        CL_low, CD_low, CM_low = _compute_low_aoa_coefficients(alpha_eff, kp, kn)
        CL_high, CD_high, CM_high = _compute_high_aoa_coefficients(
            alpha_eff, kn, deflection, params.mac, params.flap_chord
        )

        # Blending function for stall
        f_blend_pos = 0.5 * (1.0 - jnp.tanh(4.0 * (alpha_eff - alpha_max) / ALPHA_BLEND))
        f_blend_neg = 0.5 * (1.0 - jnp.tanh(-4.0 * (alpha_eff - alpha_min) / ALPHA_BLEND))
        f_blend = jnp.where(alpha_eff > 0, f_blend_pos, f_blend_neg)

        # Blended coefficients
        CL = CL_low * f_blend + CL_high * (1.0 - f_blend)
        CD = CD_low * f_blend + CD_high * (1.0 - f_blend)
        CM = CM_low * f_blend + CM_high * (1.0 - f_blend)

        # Dynamic pressure times area
        q_S = 0.5 * rho * vxz2 * params.span * params.mac

        # Forces in segment frame (wind axes to segment)
        sa = jnp.sin(alpha)
        ca = jnp.cos(alpha)
        F_S = q_S * jnp.array([
            CL * sa - CD * ca,
            0.0,
            -CL * ca - CD * sa
        ])

        # Moment in segment frame
        M_S = q_S * params.mac * jnp.array([0.0, CM, 0.0])

        # Transform to body frame
        F_body = C_BS @ F_S

        # Total moment = moment from forces + aerodynamic moment
        M_body = jnp.cross(params.position, F_body) + C_BS @ M_S

        return F_body, M_body

    # Use lax.cond for conditional
    return jax.lax.cond(vxz2 < 0.01, zero_velocity_case, normal_case)


def _compute_flap_delta_alpha(
    deflection: float,
    kp: float,
    mac: float,
    flap_chord: float,
) -> float:
    """Compute the change in zero-lift angle due to flap deflection.

    Args:
        deflection: Flap deflection [rad]
        kp: Lift curve slope
        mac: Mean aerodynamic chord [m]
        flap_chord: Flap chord [m]

    Returns:
        Change in effective alpha [rad]
    """
    eta_f, tau_f = _compute_flap_effectiveness(deflection, mac, flap_chord)
    delta_CL = kp * tau_f * eta_f * deflection
    # Approximate change in alpha for this delta CL
    # delta_CL = kp * delta_alpha, so delta_alpha = delta_CL / kp
    return delta_CL / kp


# Pre-defined segment configurations matching PX4 SIH
def create_wing_left() -> AeroSegmentParams:
    """Create left wing segment matching PX4 SIH."""
    return AeroSegmentParams(
        span=0.43,
        mac=0.21,
        alpha_0=jnp.deg2rad(-4.0),  # -4 degrees incidence
        position=jnp.array([0.0, -0.215, 0.0]),
        dihedral=jnp.deg2rad(3.0),  # 3 degrees dihedral
        aspect_ratio=4.1,
        flap_chord=0.07,  # MAC/3 for ailerons
        prop_radius=-1.0,  # No slipstream
    )


def create_wing_right() -> AeroSegmentParams:
    """Create right wing segment matching PX4 SIH."""
    return AeroSegmentParams(
        span=0.43,
        mac=0.21,
        alpha_0=jnp.deg2rad(-4.0),
        position=jnp.array([0.0, 0.215, 0.0]),
        dihedral=jnp.deg2rad(-3.0),  # Negative dihedral for right wing
        aspect_ratio=4.1,
        flap_chord=0.07,
        prop_radius=-1.0,
    )


def create_tailplane() -> AeroSegmentParams:
    """Create horizontal tailplane segment matching PX4 SIH."""
    return AeroSegmentParams(
        span=0.3,
        mac=0.1,
        alpha_0=0.0,
        position=jnp.array([-0.4, 0.0, 0.0]),
        dihedral=0.0,
        aspect_ratio=3.0,  # 0.3/0.1
        flap_chord=0.05,  # Elevator chord
        prop_radius=0.1,  # Gets propeller slipstream!
    )


def create_fin() -> AeroSegmentParams:
    """Create vertical fin segment matching PX4 SIH."""
    return AeroSegmentParams(
        span=0.25,
        mac=0.18,
        alpha_0=0.0,
        position=jnp.array([-0.45, 0.0, -0.1]),
        dihedral=jnp.deg2rad(-90.0),  # Vertical
        aspect_ratio=1.4,  # 0.25/0.18
        flap_chord=0.12,  # Rudder chord
        prop_radius=0.1,  # Gets propeller slipstream!
    )


def create_fuselage() -> AeroSegmentParams:
    """Create fuselage segment matching PX4 SIH."""
    return AeroSegmentParams(
        span=0.2,
        mac=0.8,
        alpha_0=0.0,
        position=jnp.array([0.0, 0.0, 0.0]),
        dihedral=jnp.deg2rad(-90.0),  # Vertical orientation for drag
        aspect_ratio=0.25,
        flap_chord=0.0,  # No control surface
        prop_radius=-1.0,  # No slipstream
    )

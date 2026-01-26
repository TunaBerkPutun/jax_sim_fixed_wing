"""Physical constants and aircraft properties.

Updated to match PX4 SIH fixed-wing simulator.
Reference: PX4-Autopilot/src/modules/simulation/simulator_sih/
"""

import jax.numpy as jnp

from jax_sim.physics.aero_segment import (
    create_wing_left,
    create_wing_right,
    create_tailplane,
    create_fin,
    create_fuselage,
)

# Environment constants
G = 9.81
RHO = 1.225  # Air density at sea level [kg/m^3]

# Aircraft mass
MASS = 1.0  # kg (small UAV, matches PX4 SIH_MASS)

# Inertia (Inertia Matrix)
# Match PX4 SIH defaults for fixed-wing
Ixx = 0.025
Iyy = 0.025
Izz = 0.03
Inertia = jnp.diag(jnp.array([Ixx, Iyy, Izz]))
Inertia_inv = jnp.diag(jnp.array([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz]))

# Geometry (matching PX4 SIH fixed-wing)
WING_SPAN = 0.86  # Wing span [m] (PX4: SPAN = 0.86f)
CHORD = 0.21  # Mean aerodynamic chord [m] (PX4: MAC = 0.21f)
WING_AREA = WING_SPAN * CHORD  # ~0.18 m^2
TAIL_DIST = 0.4  # Distance from CG to tail [m] (PX4: tailplane at x=-0.4)
PROP_RADIUS = 0.1  # Propeller radius [m] (PX4: RP = 0.1f)

# Thrust
T_MAX = 15.0  # Maximum thrust force [N] (increased from PX4's 5.0 for better climb)

# Servo and motor time constants (seconds)
TAU_SERVO = 0.1  # Control surface response
TAU_MOTOR = 0.05  # Match PX4 SIH thrust time constant (SIH_T_TAU = 0.05f)

# Linear damping coefficients
# NOTE: For fixed-wing, set to 0 - aerodynamic model provides all drag/damping
# KDV=1.0 and KDW=0.025 are for MULTIROTORS in PX4 SIH
KDV = 0.0  # Translational damping (0 for fixed-wing)
KDW = 0.0  # Rotational damping (0 for fixed-wing)

# Control surface limit (radians)
FLAP_MAX = jnp.deg2rad(15.0)  # PX4: FLAP_MAX = M_PI_F / 12.0f

# Body rate clamp (rad/s)
MAX_BODY_RATE = 6.0 * jnp.pi

# =============================================================================
# NEW: Aerodynamic Segment Definitions (PX4 SIH style)
# =============================================================================
# These use the full aerodynamic model with:
# - Zero-lift angle of attack (-4 deg for wings)
# - Dihedral angles
# - Propeller slipstream effects on tail
# - Proper lift/drag/moment coefficients

WING_LEFT = create_wing_left()
WING_RIGHT = create_wing_right()
TAILPLANE = create_tailplane()
FIN = create_fin()
FUSELAGE = create_fuselage()

# =============================================================================
# DEPRECATED: Old segment definitions (kept for backwards compatibility)
# =============================================================================
# [x_offset, y_offset, z_offset, Area]
# X: nose positive, Y: right positive, Z: down positive (NED)
SEGMENTS = {
    "left_wing": jnp.array([0.0, -WING_SPAN / 4.0, 0.0, WING_AREA / 2.0]),
    "right_wing": jnp.array([0.0, WING_SPAN / 4.0, 0.0, WING_AREA / 2.0]),
    "elevator": jnp.array([-TAIL_DIST, 0.0, 0.0, 0.03]),  # Area: 0.3*0.1
    "rudder": jnp.array([-TAIL_DIST - 0.05, 0.0, -0.1, 0.045]),  # Area: 0.25*0.18
}

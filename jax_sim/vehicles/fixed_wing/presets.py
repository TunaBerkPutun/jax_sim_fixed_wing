"""Fixed-wing preset airframes (spec §7.4).

This module owns:
- The numeric constants that define a specific airframe (mass, inertia,
  geometry, propulsion, servo time constants).
- The aero segment instances built from those constants.
- Factory functions that bundle them into a `FixedWingParams` pytree.

Day-one preset: PX4 SIH UAV. Future presets (different airframes) live here
alongside.
"""

import jax.numpy as jnp

from jax_sim.physics.constants import G, RHO
from jax_sim.vehicles.fixed_wing.params import (
    EnvironmentParams,
    MassProps,
    ActuatorParams,
    PropulsionParams,
    AeroSegments,
    FixedWingParams,
)
from jax_sim.vehicles.fixed_wing._aero_segment_lib import (
    create_wing_left,
    create_wing_right,
    create_tailplane,
    create_fin,
    create_fuselage,
)

# ---------------------------------------------------------------------------
# PX4 SIH UAV airframe constants
# (Reference: PX4-Autopilot/src/modules/simulation/simulator_sih/)
# ---------------------------------------------------------------------------

# Mass
MASS = 1.0  # kg, matches PX4 SIH_MASS

# Inertia (diagonal tensor for the SIH airframe)
Ixx = 0.025
Iyy = 0.025
Izz = 0.03
Inertia = jnp.diag(jnp.array([Ixx, Iyy, Izz]))
Inertia_inv = jnp.diag(jnp.array([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz]))

# Geometry
WING_SPAN = 0.86       # m, PX4: SPAN = 0.86f
CHORD = 0.21           # m, PX4: MAC = 0.21f
WING_AREA = WING_SPAN * CHORD
TAIL_DIST = 0.4        # m, tail moment arm
PROP_RADIUS = 0.1      # m, PX4: RP = 0.1f

# Propulsion
T_MAX = 15.0           # N, increased from PX4's 5.0 for better climb performance

# Actuator dynamics
TAU_SERVO = 0.1        # s, control surface first-order lag
TAU_MOTOR = 0.05       # s, PX4 SIH_T_TAU

# Limits
FLAP_MAX = jnp.deg2rad(15.0)   # rad, PX4: M_PI_F / 12.0f
MAX_BODY_RATE = 6.0 * jnp.pi   # rad/s

# Legacy multirotor damping (set to 0 for fixed-wing — aero model owns damping)
KDV = 0.0
KDW = 0.0

# ---------------------------------------------------------------------------
# Aerodynamic segment instances (PX4 SIH style)
# ---------------------------------------------------------------------------

WING_LEFT = create_wing_left()
WING_RIGHT = create_wing_right()
TAILPLANE = create_tailplane()
FIN = create_fin()
FUSELAGE = create_fuselage()

# Legacy planar segment layout (kept for backwards compatibility with older
# tooling / scripts that still reference `SEGMENTS`).
# [x_offset, y_offset, z_offset, Area] — NED body frame.
SEGMENTS = {
    "left_wing":  jnp.array([0.0, -WING_SPAN / 4.0, 0.0, WING_AREA / 2.0]),
    "right_wing": jnp.array([0.0,  WING_SPAN / 4.0, 0.0, WING_AREA / 2.0]),
    "elevator":   jnp.array([-TAIL_DIST,        0.0,  0.0, 0.03]),
    "rudder":     jnp.array([-TAIL_DIST - 0.05, 0.0, -0.1, 0.045]),
}


# ---------------------------------------------------------------------------
# Factory: PX4 SIH UAV
# ---------------------------------------------------------------------------

def create_default_fixed_wing() -> FixedWingParams:
    """Create the default fixed-wing aircraft configuration (PX4 SIH UAV)."""
    return FixedWingParams(
        environment=EnvironmentParams(gravity=G, rho=RHO),
        mass_props=MassProps(
            mass=MASS,
            inertia=Inertia,
            inertia_inv=Inertia_inv,
        ),
        actuators=ActuatorParams(
            flap_max=FLAP_MAX,
            tau_servo=TAU_SERVO,
            tau_motor=TAU_MOTOR,
            max_body_rate=MAX_BODY_RATE,
        ),
        propulsion=PropulsionParams(
            t_max=T_MAX,
            prop_radius=PROP_RADIUS,
        ),
        segments=AeroSegments(
            wing_left=WING_LEFT,
            wing_right=WING_RIGHT,
            tailplane=TAILPLANE,
            fin=FIN,
            fuselage=FUSELAGE,
        ),
    )


DEFAULT_FIXED_WING = create_default_fixed_wing()


def px4_sih_uav() -> FixedWingParams:
    """PX4 SIH UAV preset (alias of create_default_fixed_wing for naming consistency)."""
    return create_default_fixed_wing()


# Backwards-compat aliases (pre §18 restructure)
create_default_aircraft = create_default_fixed_wing
DEFAULT_AIRCRAFT = DEFAULT_FIXED_WING


__all__ = [
    # Airframe constants
    "MASS",
    "Ixx", "Iyy", "Izz", "Inertia", "Inertia_inv",
    "WING_SPAN", "CHORD", "WING_AREA", "TAIL_DIST", "PROP_RADIUS",
    "T_MAX",
    "TAU_SERVO", "TAU_MOTOR",
    "FLAP_MAX", "MAX_BODY_RATE",
    "KDV", "KDW",
    "WING_LEFT", "WING_RIGHT", "TAILPLANE", "FIN", "FUSELAGE",
    "SEGMENTS",
    # Factories
    "create_default_fixed_wing",
    "create_default_aircraft",
    "px4_sih_uav",
    "DEFAULT_FIXED_WING",
    "DEFAULT_AIRCRAFT",
]

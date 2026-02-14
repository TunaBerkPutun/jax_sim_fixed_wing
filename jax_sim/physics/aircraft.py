"""Aircraft configuration structures and defaults.

This module bundles the global physical constants and segment definitions into
structured dataclasses so callers can swap aircraft without editing code.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp

from . import constants as c
from .aero_segment import AeroSegmentParams


@flax.struct.dataclass
class EnvironmentParams:
    """Environment parameters."""
    gravity: float
    rho: float


@flax.struct.dataclass
class MassProps:
    """Mass and inertia properties."""
    mass: float
    inertia: jnp.ndarray
    inertia_inv: jnp.ndarray


@flax.struct.dataclass
class ActuatorParams:
    """Actuator and rate limits."""
    flap_max: float
    tau_servo: float
    tau_motor: float
    max_body_rate: float


@flax.struct.dataclass
class PropulsionParams:
    """Propulsion configuration."""
    t_max: float
    prop_radius: float


@flax.struct.dataclass
class AeroSegments:
    """Aerodynamic segment definitions."""
    wing_left: AeroSegmentParams
    wing_right: AeroSegmentParams
    tailplane: AeroSegmentParams
    fin: AeroSegmentParams
    fuselage: AeroSegmentParams


@flax.struct.dataclass
class AircraftParams:
    """Complete aircraft configuration for the simulator."""
    environment: EnvironmentParams
    mass_props: MassProps
    actuators: ActuatorParams
    propulsion: PropulsionParams
    segments: AeroSegments


def create_default_aircraft() -> AircraftParams:
    """Create the default fixed-wing aircraft configuration."""
    return AircraftParams(
        environment=EnvironmentParams(
            gravity=c.G,
            rho=c.RHO,
        ),
        mass_props=MassProps(
            mass=c.MASS,
            inertia=c.Inertia,
            inertia_inv=c.Inertia_inv,
        ),
        actuators=ActuatorParams(
            flap_max=c.FLAP_MAX,
            tau_servo=c.TAU_SERVO,
            tau_motor=c.TAU_MOTOR,
            max_body_rate=c.MAX_BODY_RATE,
        ),
        propulsion=PropulsionParams(
            t_max=c.T_MAX,
            prop_radius=c.PROP_RADIUS,
        ),
        segments=AeroSegments(
            wing_left=c.WING_LEFT,
            wing_right=c.WING_RIGHT,
            tailplane=c.TAILPLANE,
            fin=c.FIN,
            fuselage=c.FUSELAGE,
        ),
    )


DEFAULT_AIRCRAFT = create_default_aircraft()

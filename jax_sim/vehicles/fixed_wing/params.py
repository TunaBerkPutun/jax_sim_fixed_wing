"""Fixed-wing parameter pytrees (Vehicle Module Contract §7.1).

Pure dataclasses — no factory functions, no airframe constants. Build a
`FixedWingParams` from `presets.py::create_default_fixed_wing()` or compose
your own.

`EnvironmentParams` and `MassProps` live in `physics/dataclasses.py` because
they are vehicle-agnostic; we re-export them here for callers who want a
single import surface.
"""

from __future__ import annotations

import flax.struct

from jax_sim.physics.dataclasses import EnvironmentParams, MassProps
from jax_sim.vehicles.fixed_wing._aero_segment_lib import AeroSegmentParams


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
class FixedWingParams:
    """Complete fixed-wing aircraft configuration (Vehicle Module Contract §7.1)."""
    environment: EnvironmentParams
    mass_props: MassProps
    actuators: ActuatorParams
    propulsion: PropulsionParams
    segments: AeroSegments


# Backwards-compat alias (pre §18 restructure)
AircraftParams = FixedWingParams


__all__ = [
    "EnvironmentParams",
    "MassProps",
    "ActuatorParams",
    "PropulsionParams",
    "AeroSegments",
    "FixedWingParams",
    "AircraftParams",
    "AeroSegmentParams",
]

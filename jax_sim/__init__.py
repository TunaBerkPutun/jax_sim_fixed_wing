"""JAX-based fixed-wing UAV simulation for RL training."""

# Tier 1 fixed-wing dynamics + Vehicle Module Contract entrypoints
from jax_sim.vehicles.fixed_wing.tier1 import (
    compute_aircraft_forces_moments,
    equations_of_motion,
    get_forces_and_moments,
    update_actuators,
)
from jax_sim.vehicles.fixed_wing.simulator import simulate, simulate_batch

# Vehicle-agnostic primitives
from jax_sim.physics.rigid_body import rigid_body_step
from jax_sim.physics.constants import G, RHO
from jax_sim.physics.dataclasses import EnvironmentParams, MassProps

# Fixed-wing-specific dataclasses, airframe constants, and PX4 SIH preset
from jax_sim.vehicles.fixed_wing.params import (
    ActuatorParams,
    PropulsionParams,
    AeroSegments,
    FixedWingParams,
    AircraftParams,
)
from jax_sim.vehicles.fixed_wing.presets import (
    MASS,
    Ixx, Iyy, Izz, Inertia, Inertia_inv,
    WING_SPAN, CHORD, WING_AREA, TAIL_DIST,
    TAU_SERVO, TAU_MOTOR,
    SEGMENTS,
    create_default_fixed_wing,
    create_default_aircraft,
    DEFAULT_FIXED_WING,
    DEFAULT_AIRCRAFT,
)

__all__ = [
    "equations_of_motion",
    "get_forces_and_moments",
    "compute_aircraft_forces_moments",
    "update_actuators",
    "rigid_body_step",
    "simulate",
    "simulate_batch",
    "G", "RHO", "MASS",
    "Ixx", "Iyy", "Izz", "Inertia", "Inertia_inv",
    "WING_SPAN", "CHORD", "WING_AREA", "TAIL_DIST",
    "TAU_SERVO", "TAU_MOTOR",
    "SEGMENTS",
    "EnvironmentParams",
    "MassProps",
    "ActuatorParams",
    "PropulsionParams",
    "AeroSegments",
    "FixedWingParams",
    "AircraftParams",
    "create_default_fixed_wing",
    "create_default_aircraft",
    "DEFAULT_FIXED_WING",
    "DEFAULT_AIRCRAFT",
]

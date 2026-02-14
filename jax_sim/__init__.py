"""JAX-based fixed-wing UAV simulation for RL training."""

from jax_sim.physics.dynamics import (
    compute_aircraft_forces_moments,
    equations_of_motion,
    get_forces_and_moments,
    update_actuators,
)
from jax_sim.physics.rigid_body import rigid_body_step
from jax_sim.physics.simulator import simulate, simulate_batch
from jax_sim.physics.constants import (
    G, RHO, MASS,
    Ixx, Iyy, Izz, Inertia, Inertia_inv,
    WING_SPAN, CHORD, WING_AREA, TAIL_DIST,
    TAU_SERVO, TAU_MOTOR,
    SEGMENTS,
)
from jax_sim.physics.aircraft import (
    EnvironmentParams,
    MassProps,
    ActuatorParams,
    PropulsionParams,
    AeroSegments,
    AircraftParams,
    create_default_aircraft,
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
    "AircraftParams",
    "create_default_aircraft",
    "DEFAULT_AIRCRAFT",
]

"""JAX-based fixed-wing UAV simulation for RL training."""

from jax_sim.physics.dynamics import equations_of_motion, get_forces_and_moments
from jax_sim.physics.constants import (
    G, RHO, MASS,
    Ixx, Iyy, Izz, Inertia, Inertia_inv,
    WING_SPAN, CHORD, WING_AREA, TAIL_DIST,
    TAU_SERVO, TAU_MOTOR,
    SEGMENTS,
)

__all__ = [
    "equations_of_motion",
    "get_forces_and_moments",
    "G", "RHO", "MASS",
    "Ixx", "Iyy", "Izz", "Inertia", "Inertia_inv",
    "WING_SPAN", "CHORD", "WING_AREA", "TAIL_DIST",
    "TAU_SERVO", "TAU_MOTOR",
    "SEGMENTS",
]

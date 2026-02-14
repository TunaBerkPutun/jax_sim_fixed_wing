"""Physics simulation modules."""

from jax_sim.physics.constants import (
    G, RHO, MASS,
    Ixx, Iyy, Izz, Inertia, Inertia_inv,
    WING_SPAN, CHORD, WING_AREA, TAIL_DIST,
    TAU_SERVO, TAU_MOTOR,
    SEGMENTS,
    T_MAX, PROP_RADIUS,
    WING_LEFT, WING_RIGHT, TAILPLANE, FIN, FUSELAGE,
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
from jax_sim.physics.aerodynamics import compute_aero_segment, compute_fixed_wing_aero
from jax_sim.physics.aero_segment import (
    AeroSegmentParams,
    compute_segment_forces,
    create_wing_left,
    create_wing_right,
    create_tailplane,
    create_fin,
    create_fuselage,
)
from jax_sim.physics.dynamics import (
    compute_aircraft_forces_moments,
    equations_of_motion,
    get_forces_and_moments,
    update_actuators,
)
from jax_sim.physics.rigid_body import rigid_body_step
from jax_sim.physics.simulator import simulate, simulate_batch

__all__ = [
    # Constants
    "G", "RHO", "MASS",
    "Ixx", "Iyy", "Izz", "Inertia", "Inertia_inv",
    "WING_SPAN", "CHORD", "WING_AREA", "TAIL_DIST",
    "TAU_SERVO", "TAU_MOTOR",
    "SEGMENTS",
    "T_MAX", "PROP_RADIUS",
    # Segment definitions
    "WING_LEFT", "WING_RIGHT", "TAILPLANE", "FIN", "FUSELAGE",
    # Structured aircraft configs
    "EnvironmentParams",
    "MassProps",
    "ActuatorParams",
    "PropulsionParams",
    "AeroSegments",
    "AircraftParams",
    "create_default_aircraft",
    "DEFAULT_AIRCRAFT",
    # Aerodynamics
    "AeroSegmentParams",
    "compute_segment_forces",
    "compute_aero_segment",  # Deprecated
    "compute_fixed_wing_aero",
    # Segment factories
    "create_wing_left",
    "create_wing_right",
    "create_tailplane",
    "create_fin",
    "create_fuselage",
    # Dynamics
    "compute_aircraft_forces_moments",
    "equations_of_motion",
    "get_forces_and_moments",
    "update_actuators",
    "rigid_body_step",
    "simulate",
    "simulate_batch",
]

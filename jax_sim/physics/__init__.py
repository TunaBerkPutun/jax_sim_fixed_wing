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
from jax_sim.physics.dynamics import equations_of_motion, get_forces_and_moments

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
    "equations_of_motion",
    "get_forces_and_moments",
]

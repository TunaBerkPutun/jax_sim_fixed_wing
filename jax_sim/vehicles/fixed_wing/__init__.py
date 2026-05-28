"""Fixed-wing vehicle subpackage (spec §6, §7).

Re-exports the Vehicle Module Contract surface plus backwards-compatible
aliases for the pre-restructure names (`AircraftParams`, `DEFAULT_AIRCRAFT`,
`equations_of_motion`, etc.).
"""

# Dataclasses (params layer)
from jax_sim.vehicles.fixed_wing.params import (
    EnvironmentParams,
    MassProps,
    ActuatorParams,
    PropulsionParams,
    AeroSegments,
    FixedWingParams,
    AircraftParams,
    AeroSegmentParams,
)

# Aero segment factories
from jax_sim.vehicles.fixed_wing._aero_segment_lib import (
    compute_segment_forces,
    create_wing_left,
    create_wing_right,
    create_tailplane,
    create_fin,
    create_fuselage,
)

# PX4 SIH airframe constants and factory (presets layer)
from jax_sim.vehicles.fixed_wing.presets import (
    MASS,
    Ixx, Iyy, Izz, Inertia, Inertia_inv,
    WING_SPAN, CHORD, WING_AREA, TAIL_DIST, PROP_RADIUS,
    T_MAX,
    TAU_SERVO, TAU_MOTOR,
    FLAP_MAX, MAX_BODY_RATE,
    KDV, KDW,
    WING_LEFT, WING_RIGHT, TAILPLANE, FIN, FUSELAGE,
    SEGMENTS,
    create_default_fixed_wing,
    create_default_aircraft,
    px4_sih_uav,
    DEFAULT_FIXED_WING,
    DEFAULT_AIRCRAFT,
)

# Aero segment-model entrypoints
from jax_sim.vehicles.fixed_wing._aero_segment import (
    compute_fixed_wing_aero,
    compute_aero_segment,
)

# Tier 1 integrator and Vehicle Module Contract entrypoints (spec §7.3)
from jax_sim.vehicles.fixed_wing.tier1 import (
    init_state,
    forces_moments,
    step,
    compute_aircraft_forces_moments,
    equations_of_motion,
    get_forces_and_moments,
    update_actuators,
)

# Tier-shared helpers (trim solver, jacrev linearization)
from jax_sim.vehicles.fixed_wing._shared import (
    Tier0Coeffs,
    solve_trim,
    extract_tier0_coeffs,
)

# Tier 0 — linear-coefficient buildup (spec §4.1, §9.3). Accessible only
# under the `tier0` namespace to avoid shadowing tier1's contract names.
from jax_sim.vehicles.fixed_wing import tier0
from jax_sim.vehicles.fixed_wing.tier0 import (
    FixedWingTier0Params,
    create_default_tier0,
    DEFAULT_FIXED_WING_TIER0,
)

# High-level rollout helpers
from jax_sim.vehicles.fixed_wing.simulator import simulate, simulate_batch

__all__ = [
    # Dataclasses
    "EnvironmentParams",
    "MassProps",
    "ActuatorParams",
    "PropulsionParams",
    "AeroSegments",
    "FixedWingParams",
    "AircraftParams",
    "AeroSegmentParams",
    # Aero factories
    "compute_segment_forces",
    "create_wing_left",
    "create_wing_right",
    "create_tailplane",
    "create_fin",
    "create_fuselage",
    # Airframe constants (PX4 SIH)
    "MASS",
    "Ixx", "Iyy", "Izz", "Inertia", "Inertia_inv",
    "WING_SPAN", "CHORD", "WING_AREA", "TAIL_DIST", "PROP_RADIUS",
    "T_MAX",
    "TAU_SERVO", "TAU_MOTOR",
    "FLAP_MAX", "MAX_BODY_RATE",
    "KDV", "KDW",
    "WING_LEFT", "WING_RIGHT", "TAILPLANE", "FIN", "FUSELAGE",
    "SEGMENTS",
    # Factories / defaults
    "create_default_fixed_wing",
    "create_default_aircraft",
    "px4_sih_uav",
    "DEFAULT_FIXED_WING",
    "DEFAULT_AIRCRAFT",
    # Aero entrypoints
    "compute_fixed_wing_aero",
    "compute_aero_segment",
    # Tier 1
    "init_state",
    "forces_moments",
    "step",
    "compute_aircraft_forces_moments",
    "equations_of_motion",
    "get_forces_and_moments",
    "update_actuators",
    "simulate",
    "simulate_batch",
    # Tier 0 surface
    "tier0",
    "Tier0Coeffs",
    "FixedWingTier0Params",
    "solve_trim",
    "extract_tier0_coeffs",
    "create_default_tier0",
    "DEFAULT_FIXED_WING_TIER0",
]

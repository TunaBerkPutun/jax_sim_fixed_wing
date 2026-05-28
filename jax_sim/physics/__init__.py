"""Vehicle-agnostic physics primitives (spec §6).

After the §18 / §A.5 restructure this package contains only what every
vehicle needs: physical constants, vehicle-agnostic dataclasses, the
rigid-body integrator, and the wind model. Vehicle-specific aero, params,
and integrators live under `jax_sim.vehicles.<type>`.
"""

from jax_sim.physics.constants import G, RHO
from jax_sim.physics.dataclasses import EnvironmentParams, MassProps
from jax_sim.physics.rigid_body import rigid_body_step
from jax_sim.physics.wind import (
    WindConfig,
    create_wind_config,
    DEFAULT_WIND_CONFIG,
    one_minus_cosine_gust_scale,
    compute_gust_ned,
    update_dryden_turbulence,
    compute_total_wind_ned,
    step_wind_model,
    wind_ned_to_body,
)

__all__ = [
    # Physical constants
    "G", "RHO",
    # Vehicle-agnostic dataclasses
    "EnvironmentParams",
    "MassProps",
    # Integrator
    "rigid_body_step",
    # Wind
    "WindConfig",
    "create_wind_config",
    "DEFAULT_WIND_CONFIG",
    "one_minus_cosine_gust_scale",
    "compute_gust_ned",
    "update_dryden_turbulence",
    "compute_total_wind_ned",
    "step_wind_model",
    "wind_ned_to_body",
]

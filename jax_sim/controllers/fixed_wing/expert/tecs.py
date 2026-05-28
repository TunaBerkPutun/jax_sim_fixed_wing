"""TECS-lite — Total Energy Control System (longitudinal).

PX4 reference: `src/lib/tecs/TECS.{hpp,cpp}`. We port the essentials and skip:
- Underspeed protection (introduces a gradient cliff via `if airspeed < stall`)
- Sink-rate compensation during stall recovery
- Wind-down acceleration limiting
- Throttle slew-rate (handled at the actuator slew stage instead)
- Density compensation

What stays:
- Specific Total Energy (STE)  = g·h + 0.5·V²   → throttle
- Specific Energy Balance (SEB) = g·h·(2-W) - V·V_dot·W  → pitch
- P+I on both energy rates
- Speed weighting `W ∈ [0, 2]` (FW_T_SPDWEIGHT)
- Climb / sink rate limits
- Low-pass filters on STE/SEB rates (PX4 does this to attenuate noise; here it
  also smooths the gradient surface for our differentiable use case)

Inputs come as scalars / 1-D so the function is happy under `vmap`.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.expert.types import ExpertConfig, ExpertState
from jax_sim.physics.constants import G


@jax.jit
def tecs_step(
    altitude: jnp.ndarray,        # scalar [m] (positive up)
    airspeed: jnp.ndarray,        # scalar [m/s] (TAS)
    climb_rate: jnp.ndarray,      # scalar [m/s] (positive up — = -vel_z in NED)
    accel_x_body: jnp.ndarray,    # scalar [m/s²] (body-x acceleration ≈ V_dot)
    alt_sp: jnp.ndarray,          # scalar [m]
    tas_sp: jnp.ndarray,          # scalar [m/s]
    tecs_int_throttle: jnp.ndarray,
    tecs_int_pitch: jnp.ndarray,
    tecs_ste_rate_filt: jnp.ndarray,
    tecs_seb_rate_filt: jnp.ndarray,
    config: ExpertConfig,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One TECS-lite step.

    Returns:
        pitch_sp, throttle_sp, new_int_throttle, new_int_pitch,
        new_ste_rate_filt, new_seb_rate_filt.
    """
    # --- Climb-rate and airspeed-rate setpoints from outer errors ---------
    alt_err = alt_sp - altitude
    climb_rate_sp = jnp.clip(
        alt_err * config.tecs_alt_p,
        -config.tecs_sink_rate_max,
        config.tecs_climb_rate_max,
    )

    # Desired airspeed-rate: simple P on TAS error scaled to never exceed the
    # vertical accel budget (PX4 has a richer rule; this is the lite version).
    tas_err = tas_sp - airspeed
    tas_rate_sp = jnp.clip(
        tas_err,  # P gain 1 here; speed loop bandwidth controlled by integrators below
        -config.tecs_vert_accel_max,
        config.tecs_vert_accel_max,
    )

    # --- Specific energies ------------------------------------------------
    W = config.tecs_speed_weight  # 0..2

    # Current rates from measurements.
    ste_rate = G * climb_rate + airspeed * accel_x_body
    seb_rate = G * climb_rate * (2.0 - W) - airspeed * accel_x_body * W

    # Setpoint rates.
    ste_rate_sp = G * climb_rate_sp + airspeed * tas_rate_sp
    seb_rate_sp = G * climb_rate_sp * (2.0 - W) - airspeed * tas_rate_sp * W

    # First-order low-pass on measurements (PX4 TECS:cpp _ste_rate_estimate filter).
    alpha_ste = dt / jnp.maximum(config.tecs_ste_tc, dt)
    alpha_seb = dt / jnp.maximum(config.tecs_seb_tc, dt)
    new_ste_rate_filt = tecs_ste_rate_filt + alpha_ste * (ste_rate - tecs_ste_rate_filt)
    new_seb_rate_filt = tecs_seb_rate_filt + alpha_seb * (seb_rate - tecs_seb_rate_filt)

    # --- Errors -----------------------------------------------------------
    ste_err = ste_rate_sp - new_ste_rate_filt
    seb_err = seb_rate_sp - new_seb_rate_filt

    # --- Throttle PI on STE error ----------------------------------------
    new_int_throttle = tecs_int_throttle + config.tecs_thr_i * ste_err * dt
    # Anti-windup: clamp integrator before saturating the output.
    new_int_throttle = jnp.clip(new_int_throttle,
                                config.throttle_min - config.tecs_throttle_trim,
                                config.throttle_max - config.tecs_throttle_trim)
    throttle_sp = (config.tecs_throttle_trim
                   + config.tecs_thr_p * ste_err
                   + new_int_throttle)
    throttle_sp = jnp.clip(throttle_sp, config.throttle_min, config.throttle_max)

    # --- Pitch PI on SEB error -------------------------------------------
    new_int_pitch = tecs_int_pitch + config.tecs_pitch_i * seb_err * dt
    # Anti-windup limits per pitch bounds.
    new_int_pitch = jnp.clip(new_int_pitch,
                             config.tecs_pitch_min, config.tecs_pitch_max)
    pitch_sp = config.tecs_pitch_p * seb_err + new_int_pitch
    pitch_sp = jnp.clip(pitch_sp, config.tecs_pitch_min, config.tecs_pitch_max)

    return (pitch_sp, throttle_sp,
            new_int_throttle, new_int_pitch,
            new_ste_rate_filt, new_seb_rate_filt)

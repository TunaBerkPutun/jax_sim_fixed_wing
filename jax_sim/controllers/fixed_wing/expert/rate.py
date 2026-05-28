"""Rate loop — port of PX4 `RateControl::update` for fixed-wing.

PX4 references:
    src/lib/rate_control/rate_control.cpp:71-118      (anti-windup, FF wiring)
    src/modules/fw_rate_control/FixedwingRateControl.cpp:343-405
        (airspeed scaling, output assembly — we apply scale² to torque)

PX4's C++ uses two runtime branches per axis (saturation-aware error clamp,
landed-gate). Each becomes one `jnp.where` so the function stays jit-able and
grad-friendly. No `jax.lax.cond` is needed.

Forward pass:
    e            = rate_sp - rates                        # rad/s
    torque_pre   = Kp · e + I + FF · rate_sp              # before airspeed scale
    scale        = clip(V_trim / V, lo, hi)               # PX4 convention
    torque       = clip(torque_pre · scale², -1, 1)       # final actuator-space torque

Anti-windup (PX4 rate_control.cpp:88-118):
    sat_pos        = torque ≥ 1                            # saturation flags
    sat_neg        = torque ≤ -1
    e_eff          = e clamped so it never pushes further into saturation
    i_factor       = max(0, 1 - (e_eff / rad(400°))²)      # nonlinear attenuation
    new_I          = I + landed_gate · i_factor · Ki · e_eff · dt
    new_I          = clip(new_I, ±IMAX)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.expert.airspeed_scaling import airspeed_scale
from jax_sim.controllers.fixed_wing.expert.types import ExpertConfig

# PX4 uses 400 deg/s as the i_factor scale (rate_control.cpp:107).
_RATE_I_FACTOR_REF = jnp.deg2rad(400.0)


@jax.jit
def rate_loop(
    rate_sp: jnp.ndarray,           # (3,) commanded body rates [rad/s]
    rates: jnp.ndarray,             # (3,) measured body rates [rad/s]
    airspeed: jnp.ndarray,          # scalar TAS [m/s]
    rate_integral: jnp.ndarray,     # (3,) integrator state
    landed: jnp.ndarray,            # bool scalar (or float 0/1)
    config: ExpertConfig,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One step of the PX4-equivalent fixed-wing rate controller.

    Args:
        rate_sp: Commanded body rates [p, q, r] [rad/s].
        rates: Measured body rates [p, q, r] [rad/s].
        airspeed: True airspeed [m/s] (already filtered upstream).
        rate_integral: Per-axis integrator state.
        landed: Whether the aircraft is on the ground (gates integration).
        config: ExpertConfig with rate_kp, rate_ki, rate_ff, rate_imax,
                airspeed_trim, airspeed_scale_min/max.
        dt: Timestep [s].

    Returns:
        torques: (3,) normalized actuator-space torques in [-1, 1].
        new_integral: (3,) updated integrator state.
    """
    # --- Error & PID + FF assembly ------------------------------------------
    e = rate_sp - rates

    # Airspeed scaling (PX4 multiplies the final torque by scale²).
    scale = airspeed_scale(
        airspeed,
        config.airspeed_trim,
        config.airspeed_scale_min,
        config.airspeed_scale_max,
    )
    scale_sq = scale * scale

    # FF on setpoint, not error (PX4 rate_control.cpp:78).
    # PX4 also divides FF gains by `scale` upstream; we fold that here so the
    # config stores the "design-point" gains and the call site stays simple.
    rate_ff_scaled = config.rate_ff / jnp.maximum(scale, 1e-6)
    torque_pre = config.rate_kp * e + rate_integral + rate_ff_scaled * rate_sp
    torque_scaled = torque_pre * scale_sq

    # --- Anti-windup (PX4 rate_control.cpp:92-99, 107-108) ------------------
    # Saturation flags from the candidate (pre-clip) torque.
    sat_pos = torque_scaled >= 1.0
    sat_neg = torque_scaled <= -1.0

    # Saturation-aware error clamping: don't push further into the saturated side.
    e_eff = jnp.where(sat_pos, jnp.minimum(e, 0.0), e)
    e_eff = jnp.where(sat_neg, jnp.maximum(e_eff, 0.0), e_eff)

    # Nonlinear i_factor: quadratic taper that vanishes at ±400 deg/s error.
    i_factor = jnp.maximum(0.0, 1.0 - (e_eff / _RATE_I_FACTOR_REF) ** 2)

    # Landed gate (multiplicative, no Python branch).
    landed_gate = jnp.where(landed, 0.0, 1.0)

    # Forward Euler integration.
    new_integral = (
        rate_integral
        + landed_gate * i_factor * config.rate_ki * e_eff * dt
    )
    new_integral = jnp.clip(new_integral, -config.rate_imax, config.rate_imax)

    # --- Final clip ---------------------------------------------------------
    torques = jnp.clip(torque_scaled, -1.0, 1.0)

    return torques, new_integral

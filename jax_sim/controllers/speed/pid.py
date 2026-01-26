"""Speed controller - PI controller with feed-forward.

Converts airspeed error to throttle command.
Independent from the attitude control cascade.

Input:  Target airspeed [m/s] and current airspeed [m/s]
Output: Throttle command normalized to [0, 1]
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.pid_gains import PIDConfig, PIDState


@jax.jit
def speed_controller(
    speed_cmd: float,
    speed: float,
    pid_state: PIDState,
    config: PIDConfig,
    dt: float,
) -> Tuple[float, PIDState]:
    """PI controller with feed-forward: speed error → throttle.

    Uses feed-forward term (throttle_ff) for level flight baseline,
    plus PI correction for speed errors.

    Formula:
        error = speed_cmd - speed
        P = Kp * error
        I = Ki * integral(error) [clamped]
        throttle = throttle_ff + P + I

    Args:
        speed_cmd: Target airspeed [m/s]
        speed: Current airspeed [m/s]
        pid_state: PIDState with speed_integral
        config: PIDConfig with speed gains and throttle_ff
        dt: Timestep [seconds]

    Returns:
        throttle: Throttle command in [0, 1]
        new_pid_state: Updated PIDState with new speed_integral
    """
    # Compute speed error
    speed_error = speed_cmd - speed

    # Proportional term
    p_term = config.speed_kp * speed_error

    # Integral term with anti-windup
    new_speed_integral = pid_state.speed_integral + speed_error * dt
    new_speed_integral = jnp.clip(
        new_speed_integral,
        -config.integral_limit,
        config.integral_limit,
    )
    i_term = config.speed_ki * new_speed_integral

    # Total output with feed-forward
    throttle = config.throttle_ff + p_term + i_term

    # Clamp to valid throttle range [0, 1]
    throttle = jnp.clip(throttle, 0.0, 1.0)

    # Update state (preserve rate integrals)
    new_pid_state = PIDState(
        rate_integral=pid_state.rate_integral,
        speed_integral=new_speed_integral,
        prev_rates=pid_state.prev_rates,
    )

    return throttle, new_pid_state


@jax.jit
def speed_controller_simple(
    speed_cmd: float,
    speed: float,
    integral: float,
    kp: float,
    ki: float,
    throttle_ff: float,
    integral_limit: float,
    dt: float,
) -> Tuple[float, float]:
    """Simplified PI speed controller without PIDState.

    Useful for standalone testing or simple integrations.

    Args:
        speed_cmd: Target airspeed [m/s]
        speed: Current airspeed [m/s]
        integral: Current integral state
        kp: Proportional gain
        ki: Integral gain
        throttle_ff: Feed-forward throttle for level flight
        integral_limit: Anti-windup limit
        dt: Timestep [seconds]

    Returns:
        throttle: Throttle command in [0, 1]
        new_integral: Updated integral state
    """
    error = speed_cmd - speed

    # P term
    p_term = kp * error

    # I term with anti-windup
    new_integral = jnp.clip(
        integral + error * dt,
        -integral_limit,
        integral_limit,
    )
    i_term = ki * new_integral

    # Output with feed-forward
    throttle = jnp.clip(throttle_ff + p_term + i_term, 0.0, 1.0)

    return throttle, new_integral

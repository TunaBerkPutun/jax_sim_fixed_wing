"""Rate controller (inner loop) - PID controller.

Converts rate errors to actuator commands.
This is the inner loop of the cascade PID system.

Input:  Target rates [p_sp, q_sp, r_sp] and current rates [p, q, r] from gyro
Output: Actuator commands [aileron, elevator, rudder] normalized to [-1, 1]
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.pid_gains import PIDConfig, PIDState


@jax.jit
def rate_controller(
    rate_sp: jnp.ndarray,
    rates: jnp.ndarray,
    pid_state: PIDState,
    config: PIDConfig,
    dt: float,
) -> Tuple[jnp.ndarray, PIDState]:
    """PID controller: rate error → actuator commands.

    Uses derivative on measurement (not error) to avoid setpoint kick.
    Includes integral anti-windup via clamping.

    Formula for each axis:
        error = rate_sp - rate
        P = Kp * error
        I = Ki * integral(error) [clamped]
        D = Kd * (-d(rate)/dt)   [derivative on measurement]
        output = P + I + D

    Args:
        rate_sp: Target rates [p_sp, q_sp, r_sp] [rad/s]
        rates: Current rates [p, q, r] from gyro [rad/s]
        pid_state: PIDState with integrals and previous rates
        config: PIDConfig with gains and limits
        dt: Timestep [seconds]

    Returns:
        actuators: [aileron, elevator, rudder] commands in [-1, 1]
        new_pid_state: Updated PIDState with new integrals/prev_rates
    """
    # Compute rate errors
    rate_error = rate_sp - rates

    # Proportional term
    p_term = config.rate_kp * rate_error

    # Integral term with anti-windup
    new_rate_integral = pid_state.rate_integral + rate_error * dt
    new_rate_integral = jnp.clip(
        new_rate_integral,
        -config.integral_limit,
        config.integral_limit,
    )
    i_term = config.rate_ki * new_rate_integral

    # Derivative term on measurement (not error) to avoid setpoint kick
    # d(rate)/dt ≈ (rate - prev_rate) / dt
    # We use negative because we want to resist changes
    rate_derivative = (rates - pid_state.prev_rates) / dt
    d_term = -config.rate_kd * rate_derivative

    # Total output
    output = p_term + i_term + d_term

    # Map to actuators: [roll->aileron, pitch->elevator, yaw->rudder]
    actuators = jnp.clip(output, -1.0, 1.0)

    # Update state
    new_pid_state = PIDState(
        rate_integral=new_rate_integral,
        speed_integral=pid_state.speed_integral,
        prev_rates=rates,
    )

    return actuators, new_pid_state


@jax.jit
def rate_controller_single_axis(
    rate_sp: float,
    rate: float,
    integral: float,
    prev_rate: float,
    kp: float,
    ki: float,
    kd: float,
    integral_limit: float,
    dt: float,
) -> Tuple[float, float, float]:
    """Single-axis rate PID controller.

    Useful for debugging or when axes need different handling.

    Args:
        rate_sp: Target rate [rad/s]
        rate: Current rate [rad/s]
        integral: Current integral state
        prev_rate: Previous rate measurement
        kp, ki, kd: PID gains
        integral_limit: Anti-windup limit
        dt: Timestep [seconds]

    Returns:
        output: Actuator command [-1, 1]
        new_integral: Updated integral state
        new_prev_rate: Updated previous rate (= current rate)
    """
    error = rate_sp - rate

    # P term
    p_term = kp * error

    # I term with anti-windup
    new_integral = jnp.clip(
        integral + error * dt,
        -integral_limit,
        integral_limit,
    )
    i_term = ki * new_integral

    # D term on measurement
    d_term = -kd * (rate - prev_rate) / dt

    output = jnp.clip(p_term + i_term + d_term, -1.0, 1.0)

    return output, new_integral, rate

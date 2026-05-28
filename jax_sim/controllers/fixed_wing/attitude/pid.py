"""Attitude controller (outer loop) - P controller.

Converts attitude errors (angle errors) to rate setpoints.
This is the outer loop of the cascade PID system.

Input:  Target angles [roll_cmd, pitch_cmd] and current angles [roll, pitch]
Output: Rate setpoints [p_sp, q_sp] (angular velocities to achieve desired attitude)
"""

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.pid_gains import PIDConfig


@jax.jit
def attitude_controller(
    roll_cmd: float,
    pitch_cmd: float,
    roll: float,
    pitch: float,
    config: PIDConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """P controller: angle error → rate setpoint.

    Converts attitude errors to desired angular rates using proportional control.
    The time constant (tau) determines how aggressively the controller responds
    to attitude errors.

    Formula:
        p_sp = (roll_cmd - roll) / tau_roll
        q_sp = (pitch_cmd - pitch) / tau_pitch

    Args:
        roll_cmd: Target roll angle [rad]
        pitch_cmd: Target pitch angle [rad]
        roll: Current roll angle [rad]
        pitch: Current pitch angle [rad]
        config: PIDConfig with tau_roll, tau_pitch, and rate_limit

    Returns:
        p_sp: Roll rate setpoint [rad/s]
        q_sp: Pitch rate setpoint [rad/s]
    """
    # Compute angle errors
    roll_error = roll_cmd - roll
    pitch_error = pitch_cmd - pitch

    # Convert to rate setpoints via time constant
    # Smaller tau = faster response
    p_sp = roll_error / config.tau_roll
    q_sp = pitch_error / config.tau_pitch

    # Clamp to rate limits to prevent excessive demands
    p_sp = jnp.clip(p_sp, -config.rate_limit, config.rate_limit)
    q_sp = jnp.clip(q_sp, -config.rate_limit, config.rate_limit)

    return p_sp, q_sp


@jax.jit
def attitude_controller_vectorized(
    attitude_cmd: jnp.ndarray,
    attitude: jnp.ndarray,
    config: PIDConfig,
) -> jnp.ndarray:
    """Vectorized attitude controller for [roll, pitch].

    Args:
        attitude_cmd: Target [roll_cmd, pitch_cmd] [rad]
        attitude: Current [roll, pitch] [rad]
        config: PIDConfig

    Returns:
        rate_sp: [p_sp, q_sp] rate setpoints [rad/s]
    """
    p_sp, q_sp = attitude_controller(
        attitude_cmd[0],
        attitude_cmd[1],
        attitude[0],
        attitude[1],
        config,
    )
    return jnp.array([p_sp, q_sp])

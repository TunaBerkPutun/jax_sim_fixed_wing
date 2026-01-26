"""PID configuration and state dataclasses."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random


class PIDConfig(NamedTuple):
    """PID controller configuration.

    All gains and limits for the cascade PID system.
    Uses NamedTuple for JAX compatibility (immutable, pytree).
    """

    # Outer loop time constants (seconds)
    tau_roll: float = 0.5
    tau_pitch: float = 0.5

    # Rate PID gains [roll (p), pitch (q), yaw (r)]
    rate_kp: jnp.ndarray = jnp.array([0.5, 0.5, 0.3])
    rate_ki: jnp.ndarray = jnp.array([0.1, 0.1, 0.05])
    rate_kd: jnp.ndarray = jnp.array([0.01, 0.01, 0.005])

    # Speed PI gains
    speed_kp: float = 0.5
    speed_ki: float = 0.1
    throttle_ff: float = 0.6  # Feed-forward for level flight

    # Limits
    rate_limit: float = 1.047  # 60 deg/s in radians
    integral_limit: float = 2.0  # Anti-windup limit


class PIDState(NamedTuple):
    """PID controller state (integrals and previous measurements).

    Holds stateful values that persist between timesteps.
    Uses NamedTuple for JAX compatibility (immutable, pytree).
    """

    # Rate controller integral terms [p, q, r]
    rate_integral: jnp.ndarray = jnp.zeros(3)

    # Speed controller integral term
    speed_integral: float = 0.0

    # Previous rate measurements for derivative on measurement [p, q, r]
    prev_rates: jnp.ndarray = jnp.zeros(3)


def create_pid_config(
    tau_roll: float = 0.5,
    tau_pitch: float = 0.5,
    rate_kp: tuple = (0.5, 0.5, 0.3),
    rate_ki: tuple = (0.1, 0.1, 0.05),
    rate_kd: tuple = (0.01, 0.01, 0.005),
    speed_kp: float = 0.5,
    speed_ki: float = 0.1,
    throttle_ff: float = 0.6,
    rate_limit: float = 1.047,
    integral_limit: float = 2.0,
) -> PIDConfig:
    """Create a PIDConfig with the given parameters.

    Helper function to create PIDConfig with proper array conversion.
    """
    return PIDConfig(
        tau_roll=tau_roll,
        tau_pitch=tau_pitch,
        rate_kp=jnp.array(rate_kp),
        rate_ki=jnp.array(rate_ki),
        rate_kd=jnp.array(rate_kd),
        speed_kp=speed_kp,
        speed_ki=speed_ki,
        throttle_ff=throttle_ff,
        rate_limit=rate_limit,
        integral_limit=integral_limit,
    )


def create_pid_state() -> PIDState:
    """Create a fresh PIDState with zeroed values."""
    return PIDState(
        rate_integral=jnp.zeros(3),
        speed_integral=0.0,
        prev_rates=jnp.zeros(3),
    )


def randomize_pid_gains(
    config: PIDConfig,
    key: jax.Array,
    noise_scale: float = 0.2,
) -> PIDConfig:
    """Randomize PID gains for domain randomization.

    Multiplies each gain by a random factor in [1 - noise_scale, 1 + noise_scale].
    This helps RL agents learn robust policies that work with
    imperfectly tuned controllers.

    Args:
        config: Base PIDConfig to randomize
        key: JAX random key
        noise_scale: Maximum deviation (0.2 = ±20%)

    Returns:
        New PIDConfig with randomized gains
    """
    keys = random.split(key, 8)

    def rand_factor(k):
        """Generate random multiplier in [1-scale, 1+scale]."""
        return 1.0 + noise_scale * (2.0 * random.uniform(k) - 1.0)

    def rand_array(k, arr):
        """Randomize each element of an array."""
        subkeys = random.split(k, len(arr))
        factors = jnp.array([rand_factor(sk) for sk in subkeys])
        return arr * factors

    return PIDConfig(
        tau_roll=config.tau_roll * rand_factor(keys[0]),
        tau_pitch=config.tau_pitch * rand_factor(keys[1]),
        rate_kp=rand_array(keys[2], config.rate_kp),
        rate_ki=rand_array(keys[3], config.rate_ki),
        rate_kd=rand_array(keys[4], config.rate_kd),
        speed_kp=config.speed_kp * rand_factor(keys[5]),
        speed_ki=config.speed_ki * rand_factor(keys[6]),
        throttle_ff=config.throttle_ff * rand_factor(keys[7]),
        rate_limit=config.rate_limit,  # Don't randomize limits
        integral_limit=config.integral_limit,
    )

"""Domain randomization utilities for environment reset.

Randomizes:
- Target position and speed
- PID gains (±30% noise)
- Physics parameters (mass, servo lag)
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jax_sim.controllers.pid_gains import PIDConfig


@jax.jit
def randomize_target(key: PRNGKey) -> Tuple[jnp.ndarray, float]:
    """Generate random target position and speed.

    Args:
        key: JAX PRNG key

    Returns:
        target_pos: (3,) array in NED frame [north, east, down]
        target_speed: Scalar target airspeed [m/s]

    Target specifications:
    - Distance: 50-150m from origin
    - Direction: Random spherical (azimuth 0-360°, elevation ±45°)
    - Altitude: Clamped to [-150, -50]m (50-150m above ground)
    - Speed: 15-25 m/s
    """
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Random distance [50, 150]m
    distance = jax.random.uniform(key1, minval=50.0, maxval=150.0)

    # Random direction (spherical coordinates)
    theta = jax.random.uniform(key2, minval=0.0, maxval=2.0 * jnp.pi)  # azimuth
    phi = jax.random.uniform(key3, minval=-jnp.pi/4, maxval=jnp.pi/4)  # elevation ±45°

    # Convert spherical to Cartesian (NED frame)
    dx = distance * jnp.cos(phi) * jnp.cos(theta)  # North
    dy = distance * jnp.cos(phi) * jnp.sin(theta)  # East
    dz = distance * jnp.sin(phi)  # Down

    # Clamp altitude to safe range [-150, -50]m
    # NED: negative z is up, so -150m is 150m altitude
    dz_clamped = jnp.clip(dz, -150.0, -50.0)

    target_pos = jnp.array([dx, dy, dz_clamped])

    # Target speed [15, 25] m/s (reasonable cruise speeds)
    target_speed = jax.random.uniform(key4, minval=15.0, maxval=25.0)

    return target_pos, target_speed


@jax.jit
def randomize_pid_config(base_config: PIDConfig, key: PRNGKey) -> PIDConfig:
    """Add ±30% multiplicative noise to PID gains for robustness.

    Args:
        base_config: Nominal PID configuration (e.g., from tuned_pid_config.json)
        key: JAX PRNG key

    Returns:
        Randomized PIDConfig with noisy gains

    Note:
        - Rate limits (rate_limit, integral_limit) are kept fixed
        - All gain parameters are multiplied by (1 ± 0.3)
    """
    keys = jax.random.split(key, 8)

    # Generate multiplicative noise [0.7, 1.3] for each parameter
    def noise(k):
        return jax.random.uniform(k, minval=0.7, maxval=1.3)

    # Randomize time constants
    tau_roll = base_config.tau_roll * noise(keys[0])
    tau_pitch = base_config.tau_pitch * noise(keys[1])

    # Randomize rate PID gains (element-wise)
    rate_kp_noise = jax.random.uniform(keys[2], shape=(3,), minval=0.7, maxval=1.3)
    rate_ki_noise = jax.random.uniform(keys[3], shape=(3,), minval=0.7, maxval=1.3)
    rate_kd_noise = jax.random.uniform(keys[4], shape=(3,), minval=0.7, maxval=1.3)

    rate_kp = base_config.rate_kp * rate_kp_noise
    rate_ki = base_config.rate_ki * rate_ki_noise
    rate_kd = base_config.rate_kd * rate_kd_noise

    # Randomize speed PI gains
    speed_kp = base_config.speed_kp * noise(keys[5])
    speed_ki = base_config.speed_ki * noise(keys[6])
    throttle_ff = base_config.throttle_ff * noise(keys[7])

    # Keep limits fixed (no randomization)
    return PIDConfig(
        tau_roll=tau_roll,
        tau_pitch=tau_pitch,
        rate_kp=rate_kp,
        rate_ki=rate_ki,
        rate_kd=rate_kd,
        speed_kp=speed_kp,
        speed_ki=speed_ki,
        throttle_ff=throttle_ff,
        rate_limit=base_config.rate_limit,  # Fixed
        integral_limit=base_config.integral_limit,  # Fixed
    )


@jax.jit
def randomize_physics_params(key: PRNGKey) -> dict:
    """Randomize physics parameters for robustness.

    Args:
        key: JAX PRNG key

    Returns:
        Dictionary with:
            - mass_mult: Mass multiplier [0.8, 1.2] (±20%)
            - servo_lag: Servo time constant [0.05, 0.15]s

    Note:
        These parameters are not directly used in the current physics model
        but are provided for future extension. The servo lag could be used
        to override TAU_SERVO in dynamics.py if domain randomization is added there.
    """
    key1, key2 = jax.random.split(key)

    # Mass variation ±20%
    mass_mult = jax.random.uniform(key1, minval=0.8, maxval=1.2)

    # Servo lag (actuator dynamics time constant)
    servo_lag = jax.random.uniform(key2, minval=0.05, maxval=0.15)

    return {
        "mass_mult": mass_mult,
        "servo_lag": servo_lag,
    }

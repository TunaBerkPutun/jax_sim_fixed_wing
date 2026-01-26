"""Episode termination conditions.

Checks for success, crash, out-of-bounds, and timeout.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


# Episode limits
MAX_TIME = 30.0  # seconds (shorter episodes for faster learning)
SUCCESS_DISTANCE = 10.0  # meters (more forgiving)
SUCCESS_SPEED_ERROR = 3.0  # m/s (more forgiving)
OUT_OF_BOUNDS_DISTANCE = 300.0  # meters (must be > target distance 150m)


@jax.jit
def check_termination(
    plane_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_speed: float,
    time: float,
) -> Tuple[bool, float, Dict[str, bool]]:
    """Check if episode should terminate and compute terminal reward.

    Args:
        plane_state: (17,) aircraft state [pos(3), vel(3), quat(4), omega(3), actuators(4)]
        target_pos: (3,) target position in NED frame
        target_speed: Scalar target airspeed [m/s]
        time: Elapsed time in episode [seconds]

    Returns:
        done: Boolean indicating if episode should terminate
        terminal_reward: Additional reward for terminal state (0 if not done)
        info: Dictionary with termination reasons

    Termination conditions (checked in order of priority):
        1. Crash: altitude > 0 (hit ground in NED frame) → -2000
        2. Success: distance < 5m AND speed_error < 2 m/s → +2000
        3. Out of bounds: distance > 100m → -500
        4. Timeout: time >= 50s → 0
    """
    # Extract state
    pos = plane_state[0:3]
    vel = plane_state[3:6]

    # Compute distances and speeds
    distance = jnp.linalg.norm(target_pos - pos)
    speed = jnp.linalg.norm(vel)
    speed_error = jnp.abs(target_speed - speed)

    # NED frame: z > 0 means below ground (crash)
    altitude = -pos[2]  # Convert NED z to altitude (positive up)
    crash = pos[2] > 0.0  # Crashed if z > 0

    # Success: close to target with correct speed
    success = (distance < SUCCESS_DISTANCE) & (speed_error < SUCCESS_SPEED_ERROR)

    # Out of bounds: too far from target
    oob = distance > OUT_OF_BOUNDS_DISTANCE

    # Timeout: exceeded max episode length
    timeout = time >= MAX_TIME

    # Episode done if any termination condition is met
    done = crash | success | oob | timeout

    # Terminal rewards (mutually exclusive, priority order)
    # Reduced magnitudes to not dominate step rewards
    terminal_reward = jnp.where(
        crash, -50.0,
        jnp.where(
            success, 100.0,
            jnp.where(
                oob, -20.0,
                0.0  # timeout or no termination
            )
        )
    )

    # Info dictionary
    info = {
        "success": success,
        "crash": crash,
        "oob": oob,
        "timeout": timeout,
        "distance": distance,
        "altitude": altitude,
        "speed_error": speed_error,
    }

    return done, terminal_reward, info

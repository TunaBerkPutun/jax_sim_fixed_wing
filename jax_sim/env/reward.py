"""Reward function for trajectory tracking task.

Combines distance penalty, speed tracking, alignment bonus, and smoothness penalty.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


@jax.jit
def compute_reward(
    plane_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_speed: float,
    action: jnp.ndarray,
    last_action: jnp.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Compute reward for current state and action.

    Args:
        plane_state: (17,) aircraft state [pos(3), vel(3), quat(4), omega(3), actuators(4)]
        target_pos: (3,) target position in NED frame
        target_speed: Scalar target airspeed [m/s]
        action: (4,) current action [roll_sp, pitch_sp, yaw_rate_sp, speed_delta]
        last_action: (4,) previous action (for smoothness)

    Returns:
        reward: Scalar reward value
        info: Dictionary with reward components for logging

    Reward components:
        1. Distance penalty: -0.01 * log(1 + distance)
           - Log scale makes reward decrease rapidly when close
           - Always negative, approaches 0 as distance → 0

        2. Speed tracking: -0.5 * |speed_error|
           - Linear penalty for speed deviation
           - Encourages matching target speed

        3. Alignment bonus: +0.1 * dot(vel_dir, target_dir) if positive
           - Rewards flying toward target
           - Only positive when heading toward target

        4. Smoothness penalty: -0.01 * ||action - last_action||²
           - Discourages jerky control inputs
           - Encourages smooth trajectories
    """
    # Extract relevant quantities
    pos = plane_state[0:3]
    vel = plane_state[3:6]

    # 0. Alive bonus (encourages survival)
    alive_bonus = 1.0

    # 1. Distance to target - reward for getting closer
    distance = jnp.linalg.norm(target_pos - pos)
    # Normalized distance reward: 1.0 at distance=0, 0 at distance=100
    dist_reward = jnp.maximum(0.0, 1.0 - distance / 100.0)

    # 2. Speed tracking (reduced penalty)
    speed = jnp.linalg.norm(vel)
    speed_error = jnp.abs(target_speed - speed)
    speed_reward = -0.1 * speed_error  # Reduced from -0.5

    # 3. Alignment bonus (velocity direction vs target direction)
    vel_dir = vel / (speed + 1e-6)  # Normalize velocity, avoid division by zero
    target_dir = (target_pos - pos) / (distance + 1e-6)  # Direction to target
    alignment = jnp.dot(vel_dir, target_dir)

    # Reward alignment more strongly
    alignment_reward = jnp.where(alignment > 0.0, 1.0 * alignment, -0.5 * alignment)

    # 4. Smoothness penalty (action changes)
    action_diff = action - last_action
    smoothness_penalty = -0.01 * jnp.sum(action_diff ** 2)

    # Total reward (should be roughly -2 to +4 per step)
    total_reward = alive_bonus + dist_reward + speed_reward + alignment_reward + smoothness_penalty

    # Info dictionary for logging and analysis
    info = {
        "distance": distance,
        "speed": speed,
        "speed_error": speed_error,
        "alignment": alignment,
        "reward/alive": alive_bonus,
        "reward/distance": dist_reward,
        "reward/speed": speed_reward,
        "reward/alignment": alignment_reward,
        "reward/smoothness": smoothness_penalty,
        "reward/total": total_reward,
    }

    return total_reward, info

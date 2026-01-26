"""Observation extraction and normalization.

Produces a 19D normalized observation vector from environment state.
"""

import jax
import jax.numpy as jnp

from jax_sim.utils.quaternion import quat_inv, rotate_vec_by_quat


@jax.jit
def get_observation(
    plane_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_speed: float,
    last_action: jnp.ndarray,
) -> jnp.ndarray:
    """Extract 19D normalized observation from environment state.

    Args:
        plane_state: (17,) aircraft state [pos(3), vel(3), quat(4), omega(3), actuators(4)]
        target_pos: (3,) target position in NED frame
        target_speed: Scalar target airspeed [m/s]
        last_action: (4,) previous action [roll, pitch, yaw_rate, speed_delta]

    Returns:
        obs: (19,) normalized observation vector

    Observation components (19D):
        [0:3]   Relative target position in body frame (normalized by 100m)
        [3:6]   Body velocity (normalized by 30 m/s)
        [6]     Speed error (normalized by 20 m/s)
        [7:11]  Quaternion (already normalized)
        [11:14] Angular rates (normalized by 2 rad/s)
        [14:18] Actuator states (already [-1, 1])
        [18]    Last speed delta action (already [-1, 1])
    """
    # Unpack state
    pos = plane_state[0:3]  # NED position
    vel = plane_state[3:6]  # NED velocity
    quat = plane_state[6:10]  # Quaternion [w, x, y, z]
    omega = plane_state[10:13]  # Body rates [p, q, r]
    actuators = plane_state[13:17]  # [aileron, elevator, rudder, throttle]

    # 1. Relative target position in BODY frame (3D)
    # Transform from NED to body frame using inverse quaternion
    rel_pos_ned = target_pos - pos
    quat_inv_val = quat_inv(quat)
    rel_pos_body = rotate_vec_by_quat(quat_inv_val, rel_pos_ned)
    rel_pos_norm = rel_pos_body / 100.0  # Normalize by max expected distance

    # 2. Body velocity (3D)
    # Convert NED velocity to body frame
    vel_body = rotate_vec_by_quat(quat_inv_val, vel)
    vel_norm = vel_body / 30.0  # Normalize by max speed

    # 3. Speed error (1D)
    speed = jnp.linalg.norm(vel)
    speed_error = target_speed - speed
    speed_error_norm = speed_error / 20.0  # Normalize by reasonable speed range

    # 4. Quaternion (4D) - already normalized by definition
    quat_obs = quat

    # 5. Angular rates (3D)
    omega_norm = omega / 2.0  # Normalize by max expected rate (~2 rad/s)

    # 6. Actuator states (4D) - already in [-1, 1]
    actuator_obs = actuators

    # 7. Last speed action (1D) - only include speed delta component
    # Last action is [roll_sp, pitch_sp, yaw_rate_sp, speed_delta]
    last_speed_action = last_action[3]  # Already in [-1, 1]

    # Concatenate all observations (19D total)
    obs = jnp.concatenate([
        rel_pos_norm,       # [0:3]   (3D)
        vel_norm,           # [3:6]   (3D)
        speed_error_norm[None],  # [6]     (1D) - add dimension
        quat_obs,           # [7:11]  (4D)
        omega_norm,         # [11:14] (3D)
        actuator_obs,       # [14:18] (4D)
        last_speed_action[None],  # [18]    (1D) - add dimension
    ])

    return obs

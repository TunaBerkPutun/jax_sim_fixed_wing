"""Quaternion math utilities."""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def rotate_vec_by_quat(q, v):
    """Rotate vector by quaternion (Body -> Earth or inverse).

    Args:
        q: Quaternion [w, x, y, z]
        v: Vector to rotate [x, y, z]

    Returns:
        Rotated vector [x, y, z]
    """
    q_vec = q[1:]
    a = q[0]
    # Rodrigues-like rotation formula
    t = 2.0 * jnp.cross(q_vec, v)
    return v + a * t + jnp.cross(q_vec, t)


@jax.jit
def quat_inv(q):
    """Quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Inverse quaternion [w, -x, -y, -z]
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


@jax.jit
def quat_derivative(q, omega):
    """Quaternion derivative: dq/dt = 0.5 * q * omega.

    Args:
        q: Quaternion [w, x, y, z]
        omega: Angular velocity [p, q, r] (body rates)

    Returns:
        Quaternion derivative [dw, dx, dy, dz]
    """
    w, x, y, z = q
    p, q_val, r = omega

    dw = -0.5 * (x * p + y * q_val + z * r)
    dx = 0.5 * (w * p + y * r - z * q_val)
    dy = 0.5 * (w * q_val - x * r + z * p)
    dz = 0.5 * (w * r + x * q_val - y * p)

    return jnp.array([dw, dx, dy, dz])


# Gimbal lock protection: ±85 degrees
_PITCH_LIMIT = 85.0 * jnp.pi / 180.0


@jax.jit
def quat_to_euler_jax(quat):
    """Convert quaternion to Euler angles (JAX-compatible).

    Aerospace sequence: roll (x), pitch (y), yaw (z).
    Includes gimbal lock protection by clamping pitch to ±85°.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Array [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = jnp.arctan2(t0, t1)

    # Pitch (y-axis rotation) with gimbal lock protection
    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = jnp.clip(t2, -1.0, 1.0)
    pitch = jnp.arcsin(t2)
    pitch = jnp.clip(pitch, -_PITCH_LIMIT, _PITCH_LIMIT)

    # Yaw (z-axis rotation)
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = jnp.arctan2(t3, t4)

    return jnp.array([roll, pitch, yaw])


def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Aerospace sequence: roll (x), pitch (y), yaw (z).
    Works with numpy arrays for batch processing.

    Args:
        qw, qx, qy, qz: Quaternion components (can be arrays)

    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix (Body to Earth).

    Args:
        qw, qx, qy, qz: Quaternion components

    Returns:
        3x3 rotation matrix
    """
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

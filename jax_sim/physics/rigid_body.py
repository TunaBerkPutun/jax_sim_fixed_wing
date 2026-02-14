"""Rigid-body dynamics integration utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_sim.physics.aircraft import EnvironmentParams, MassProps
from jax_sim.utils.quaternion import rotate_vec_by_quat


@jax.jit
def rigid_body_step(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    quat: jnp.ndarray,
    omega: jnp.ndarray,
    F_body: jnp.ndarray,
    M_body: jnp.ndarray,
    dt: float,
    mass_props: MassProps,
    environment: EnvironmentParams,
    max_body_rate: float,
):
    """Integrate rigid-body state forward one timestep.

    Args:
        pos: Position in earth frame [m]
        vel: Velocity in earth frame [m/s]
        quat: Body-to-earth quaternion
        omega: Body angular velocity [rad/s]
        F_body: Total force in body frame [N]
        M_body: Total moment in body frame [Nm]
        dt: Timestep [s]
        mass_props: Mass and inertia parameters
        environment: Environment parameters (gravity)
        max_body_rate: Body rate clamp [rad/s]

    Returns:
        Tuple of (pos, vel, quat, omega) at next timestep.
    """
    # Linear acceleration (Newton F=ma)
    F_earth = rotate_vec_by_quat(quat, F_body)
    F_gravity = jnp.array([0.0, 0.0, mass_props.mass * environment.gravity])
    accel_earth = (F_earth + F_gravity) / mass_props.mass

    # Angular acceleration (Euler's rotation equations)
    term_gyroscopic = jnp.cross(omega, mass_props.inertia @ omega)
    angular_accel = mass_props.inertia_inv @ (M_body - term_gyroscopic)

    # Integration (Euler method)
    new_pos = pos + vel * dt
    new_vel = vel + accel_earth * dt
    new_omega = omega + angular_accel * dt
    new_omega = jnp.clip(new_omega, -max_body_rate, max_body_rate)

    # Quaternion integration via axis-angle
    omega_norm = jnp.linalg.norm(new_omega)
    angle = omega_norm * dt
    half_angle = 0.5 * angle
    safe_norm = jnp.where(omega_norm > 1e-8, omega_norm, 1.0)
    axis = new_omega / safe_norm
    axis = jnp.where(omega_norm > 1e-8, axis, jnp.zeros(3))
    dq = jnp.concatenate([
        jnp.array([jnp.cos(half_angle)]),
        axis * jnp.sin(half_angle),
    ])
    w1, x1, y1, z1 = quat
    w2, x2, y2, z2 = dq
    new_quat = jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    quat_norm = jnp.linalg.norm(new_quat)
    quat_norm = jnp.maximum(quat_norm, 1e-8)
    new_quat = new_quat / quat_norm

    return new_pos, new_vel, new_quat, new_omega

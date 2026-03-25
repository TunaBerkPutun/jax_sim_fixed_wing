"""Rigid-body dynamics integration utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_sim.physics.aircraft import EnvironmentParams, MassProps
from jax_sim.utils.quaternion import quat_derivative, rotate_vec_by_quat


@jax.jit
def _normalize_quat(quat: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion with numerical safety."""
    quat_norm = jnp.linalg.norm(quat)
    quat_norm = jnp.maximum(quat_norm, 1e-8)
    return quat / quat_norm


@jax.jit
def _rigid_body_derivative(
    state: jnp.ndarray,
    F_body: jnp.ndarray,
    M_body: jnp.ndarray,
    mass_props: MassProps,
    environment: EnvironmentParams,
) -> jnp.ndarray:
    """Compute continuous-time rigid-body state derivative.

    State ordering: [pos(3), vel(3), quat(4), omega(3)].
    """
    vel = state[3:6]
    quat = _normalize_quat(state[6:10])
    omega = state[10:13]

    # Linear acceleration (Newton F=ma)
    F_earth = rotate_vec_by_quat(quat, F_body)
    F_gravity = jnp.array([0.0, 0.0, mass_props.mass * environment.gravity])
    accel_earth = (F_earth + F_gravity) / mass_props.mass

    # Angular acceleration (Euler's rotation equations)
    term_gyroscopic = jnp.cross(omega, mass_props.inertia @ omega)
    angular_accel = mass_props.inertia_inv @ (M_body - term_gyroscopic)

    dpos = vel
    dvel = accel_earth
    dquat = quat_derivative(quat, omega)
    domega = angular_accel

    return jnp.concatenate([dpos, dvel, dquat, domega])


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
    # RK4 integration for state [pos, vel, quat, omega]
    state = jnp.concatenate([pos, vel, _normalize_quat(quat), omega])
    k1 = _rigid_body_derivative(state, F_body, M_body, mass_props, environment)
    k2 = _rigid_body_derivative(state + 0.5 * dt * k1, F_body, M_body, mass_props, environment)
    k3 = _rigid_body_derivative(state + 0.5 * dt * k2, F_body, M_body, mass_props, environment)
    k4 = _rigid_body_derivative(state + dt * k3, F_body, M_body, mass_props, environment)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_pos = new_state[0:3]
    new_vel = new_state[3:6]
    new_quat = _normalize_quat(new_state[6:10])
    new_omega = jnp.clip(new_state[10:13], -max_body_rate, max_body_rate)

    return new_pos, new_vel, new_quat, new_omega

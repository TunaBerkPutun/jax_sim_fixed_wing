"""Simulation dynamics (equations of motion).

Updated to use PX4 SIH-style aerodynamic model with proper:
- Zero-lift angle of attack for wings
- Propeller slipstream on tail surfaces
- Pitching moment calculation
- Correct elevator sign convention
"""

import jax
import jax.numpy as jnp

from jax_sim.physics.constants import (
    G,
    MASS,
    Inertia,
    Inertia_inv,
    TAU_MOTOR,
    FLAP_MAX,
    MAX_BODY_RATE,
    T_MAX,
)
from jax_sim.physics.aerodynamics import compute_fixed_wing_aero
from jax_sim.utils.quaternion import rotate_vec_by_quat, quat_inv


@jax.jit
def get_forces_and_moments(state, current_actuators):
    """Compute total aerodynamic forces and moments.

    Uses PX4 SIH-style aerodynamic segment model with:
    - Wing segments with -4 deg incidence and dihedral
    - Tailplane with propeller slipstream
    - Vertical fin with propeller slipstream
    - Fuselage drag
    - Correct elevator sign convention (negated internally)

    Args:
        state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r, ...]
        current_actuators: [aileron, elevator, rudder, throttle]
                          - aileron/elevator/rudder in radians
                          - throttle in 0-1

    Returns:
        Total_F_body: Total force in body frame [Fx, Fy, Fz] [N]
        Total_M_body: Total moment in body frame [Mx, My, Mz] [Nm]
        v_body: Velocity in body frame [u, v, w] [m/s]
    """
    # State unpack
    omega = state[10:13]  # p, q, r [rad/s]
    altitude = -state[2]  # Altitude (NED: -z is up)

    # Convert earth-frame velocity to body frame (v_body)
    quat_inv_val = quat_inv(state[6:10])
    v_body = rotate_vec_by_quat(quat_inv_val, state[3:6])

    # Actuators (radians for surfaces, 0-1 for throttle)
    # Convert radians back to normalized [-1, 1] for new aero model
    aileron_norm = current_actuators[0] / FLAP_MAX
    elevator_norm = current_actuators[1] / FLAP_MAX
    rudder_norm = current_actuators[2] / FLAP_MAX
    throttle = current_actuators[3]

    # Compute aerodynamic forces and moments using full segment model
    # This includes all segments (wings, tail, fin, fuselage) and
    # handles elevator sign convention internally
    F_aero, M_aero = compute_fixed_wing_aero(
        v_body=v_body,
        omega=omega,
        aileron=aileron_norm,
        elevator=elevator_norm,
        rudder=rudder_norm,
        throttle=throttle,
        altitude=altitude,
    )

    # Thrust force (body frame, forward)
    Thrust_Force = jnp.array([throttle * T_MAX, 0.0, 0.0])

    # Total force and moment (body frame)
    # Note: No KDV/KDW damping - handled by aerodynamic model
    Total_F_body = F_aero + Thrust_Force
    Total_M_body = M_aero

    return Total_F_body, Total_M_body, v_body


@jax.jit
def equations_of_motion(state, user_commands, dt=0.004):
    """Simulate one timestep of the aircraft dynamics.

    Args:
        state: [pos(3), vel(3), quat(4), omega(3), actuators(4)] -> size: 17
        user_commands: [ail_cmd, ele_cmd, rud_cmd, thr_cmd] (-1 to 1)
        dt: Timestep (seconds)

    Returns:
        next_state: Updated state vector
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    current_actuators = state[13:17]

    # Compute forces
    F_body, M_body, v_body = get_forces_and_moments(state, current_actuators)

    # 1. Linear acceleration (Newton F=ma)
    # Rotate F_body to earth frame and add gravity
    F_earth = rotate_vec_by_quat(quat, F_body)
    F_gravity = jnp.array([0.0, 0.0, MASS * G])  # NED: down is +Z

    accel_earth = (F_earth + F_gravity) / MASS

    # 2. Angular acceleration (Euler's rotation equations)
    # I * dw/dt + w x (I * w) = M
    # dw/dt = I_inv * (M - w x (I * w))
    term_gyroscopic = jnp.cross(omega, Inertia @ omega)
    angular_accel = Inertia_inv @ (M_body - term_gyroscopic)

    # 3. Integration (Euler method - fast enough for RL)
    # Position
    new_pos = pos + vel * dt
    # Velocity
    new_vel = vel + accel_earth * dt
    # Angular rate
    new_omega = omega + angular_accel * dt
    # Clamp body rates (PX4 SIH)
    new_omega = jnp.clip(new_omega, -MAX_BODY_RATE, MAX_BODY_RATE)

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
    # Quaternion multiply: q_new = q * dq
    w1, x1, y1, z1 = quat
    w2, x2, y2, z2 = dq
    new_quat = jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    # Normalize quaternion with NaN guard
    quat_norm = jnp.linalg.norm(new_quat)
    quat_norm = jnp.maximum(quat_norm, 1e-8)  # Prevent division by zero
    new_quat = new_quat / quat_norm

    # Map commands to physical limits
    target_actuators = jnp.array(
        [
            user_commands[0] * FLAP_MAX,  # Aileron
            user_commands[1] * FLAP_MAX,  # Elevator
            user_commands[2] * FLAP_MAX,  # Rudder
            jnp.clip(user_commands[3], 0.0, 1.0),  # Throttle
        ]
    )
    # Fixed-wing behavior: immediate control surfaces, lagged throttle
    alpha_lag = dt / TAU_MOTOR
    throttle = current_actuators[3] + alpha_lag * (target_actuators[3] - current_actuators[3])
    new_actuators = jnp.concatenate([target_actuators[:3], jnp.array([throttle])])

    # Pack state
    next_state = jnp.concatenate([new_pos, new_vel, new_quat, new_omega, new_actuators])

    # Simple ground contact check
    # Z > 0 is below ground in NED; we visualize altitude as -Z.
    # Z = 0 is ground, Z < 0 is above ground.
    landed = new_pos[2] >= 0.0

    # If crashed, stop motion (simple reset logic)
    next_state = jax.lax.select(
        landed,
        jnp.concatenate([new_pos, jnp.zeros(3), new_quat, jnp.zeros(3), jnp.zeros(4)]),
        next_state,
    )

    return next_state

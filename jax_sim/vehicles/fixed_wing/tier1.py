"""Fixed-wing Tier 1 dynamics (segment-based aero, current-gen physics).

PX4-SIH-style segment aero + RK4 rigid body integration. The Vehicle Module
Contract §7.3 entrypoints (`init_state`, `forces_moments`, `step`) are defined
at the bottom of this file.
"""

import jax
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing.params import FixedWingParams
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_FIXED_WING
from jax_sim.vehicles.fixed_wing._aero_segment import compute_fixed_wing_aero
from jax_sim.physics.rigid_body import rigid_body_step
from jax_sim.utils.quaternion import rotate_vec_by_quat, quat_inv

AircraftParams = FixedWingParams
DEFAULT_AIRCRAFT = DEFAULT_FIXED_WING


@jax.jit
def compute_aircraft_forces_moments(
    state,
    current_actuators,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
    wind_body: jnp.ndarray = jnp.zeros(3),
):
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
        aircraft: Aircraft configuration (segments, limits, propulsion)

    Returns:
        Total_F_body: Total force in body frame [Fx, Fy, Fz] [N]
        Total_M_body: Total moment in body frame [Mx, My, Mz] [Nm]
        v_body: Air-relative velocity in body frame [u, v, w] [m/s]
    """
    # State unpack
    omega = state[10:13]  # p, q, r [rad/s]
    altitude = -state[2]  # Altitude (NED: -z is up)

    # Convert earth-frame velocity to body frame (v_body)
    quat_inv_val = quat_inv(state[6:10])
    v_body_ground = rotate_vec_by_quat(quat_inv_val, state[3:6])
    v_body = v_body_ground - wind_body

    # Actuators (radians for surfaces, 0-1 for throttle)
    # Convert radians back to normalized [-1, 1] for new aero model
    flap_max = aircraft.actuators.flap_max
    aileron_norm = current_actuators[0] / flap_max
    elevator_norm = current_actuators[1] / flap_max
    rudder_norm = current_actuators[2] / flap_max
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
        aircraft=aircraft,
    )

    # Thrust force (body frame, forward)
    Thrust_Force = jnp.array([throttle * aircraft.propulsion.t_max, 0.0, 0.0])

    # Total force and moment (body frame)
    # Note: No KDV/KDW damping - handled by aerodynamic model
    Total_F_body = F_aero + Thrust_Force
    Total_M_body = M_aero

    return Total_F_body, Total_M_body, v_body


@jax.jit
def get_forces_and_moments(
    state,
    current_actuators,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
    wind_body: jnp.ndarray = jnp.zeros(3),
):
    """Backwards-compatible wrapper for aircraft-specific forces/moments."""
    return compute_aircraft_forces_moments(state, current_actuators, aircraft, wind_body)


@jax.jit
def update_actuators(
    current_actuators: jnp.ndarray,
    user_commands: jnp.ndarray,
    dt: float,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
) -> jnp.ndarray:
    """Map user commands to actuator states with limits and lag."""
    flap_max = aircraft.actuators.flap_max
    target_actuators = jnp.array(
        [
            user_commands[0] * flap_max,  # Aileron
            user_commands[1] * flap_max,  # Elevator
            user_commands[2] * flap_max,  # Rudder
            jnp.clip(user_commands[3], 0.0, 1.0),  # Throttle
        ]
    )
    alpha_servo = dt / aircraft.actuators.tau_servo
    alpha_motor = dt / aircraft.actuators.tau_motor
    surfaces = current_actuators[:3] + alpha_servo * (target_actuators[:3] - current_actuators[:3])
    throttle = current_actuators[3] + alpha_motor * (target_actuators[3] - current_actuators[3])
    return jnp.concatenate([surfaces, jnp.array([throttle])])


@jax.jit
def equations_of_motion(
    state,
    user_commands,
    dt: float = 0.004,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
    wind_body: jnp.ndarray = jnp.zeros(3),
):
    """Simulate one timestep of the aircraft dynamics.

    Args:
        state: [pos(3), vel(3), quat(4), omega(3), actuators(4)] -> size: 17
        user_commands: [ail_cmd, ele_cmd, rud_cmd, thr_cmd] (-1 to 1)
        dt: Timestep (seconds)
        aircraft: Aircraft configuration (mass, inertia, limits)
        wind_body: Wind velocity in body frame [u, v, w] [m/s]

    Returns:
        next_state: Updated state vector
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    current_actuators = state[13:17]

    # Compute forces
    F_body, M_body, v_body = compute_aircraft_forces_moments(
        state,
        current_actuators,
        aircraft=aircraft,
        wind_body=wind_body,
    )

    # Rigid-body integration (physics-only)
    new_pos, new_vel, new_quat, new_omega = rigid_body_step(
        pos=pos,
        vel=vel,
        quat=quat,
        omega=omega,
        F_body=F_body,
        M_body=M_body,
        dt=dt,
        mass_props=aircraft.mass_props,
        environment=aircraft.environment,
        max_body_rate=aircraft.actuators.max_body_rate,
    )

    # Aircraft-specific actuator dynamics
    new_actuators = update_actuators(
        current_actuators=current_actuators,
        user_commands=user_commands,
        dt=dt,
        aircraft=aircraft,
    )

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


# ---------------------------------------------------------------------------
# Vehicle Module Contract entrypoints (spec §7.3)
# ---------------------------------------------------------------------------

def init_state(
    pos: jnp.ndarray = jnp.zeros(3),
    vel: jnp.ndarray = jnp.array([20.0, 0.0, 0.0]),
    quat: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0]),
    omega: jnp.ndarray = jnp.zeros(3),
    actuators: jnp.ndarray = jnp.zeros(4),
) -> jnp.ndarray:
    """Build a valid (17,) fixed-wing state from initial conditions."""
    return jnp.concatenate([pos, vel, quat, omega, actuators])


@jax.jit
def forces_moments(
    state: jnp.ndarray,
    params: FixedWingParams = DEFAULT_FIXED_WING,
    wind_body: jnp.ndarray = jnp.zeros(3),
):
    """Return (F_body, M_body). Side-export for jacrev-based Tier 0 derivation."""
    F, M, _ = compute_aircraft_forces_moments(state, state[13:17], params, wind_body)
    return F, M


# `step` matches the spec's uniform signature: (state, cmd, dt, params, wind_body)
step = equations_of_motion

"""Cascade PID controller - main entry point.

Combines attitude (outer loop), rate (inner loop), and speed controllers
into a single step function that converts high-level setpoints to actuator commands.

This module acts as the "Mock Autopilot" sitting between the RL agent and physics.

Architecture:
    RL Agent → [Cascade PID] → Physics Engine

    setpoints [φ_cmd, θ_cmd, r_cmd, V_cmd]
              ↓
    ┌─────────────────────────────────────┐
    │  1. Quaternion → Euler              │
    │  2. Attitude Controller (P)         │  Outer Loop
    │     angle error → rate setpoint     │
    │  3. Rate Controller (PID)           │  Inner Loop
    │     rate error → actuator           │
    │  4. Speed Controller (PI)           │  Parallel
    │     speed error → throttle          │
    └─────────────────────────────────────┘
              ↓
    actuators [aileron, elevator, rudder, throttle]
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.utils.quaternion import quat_to_euler_jax
from jax_sim.controllers.fixed_wing.pid_gains import PIDConfig, PIDState, create_pid_state
from jax_sim.controllers.fixed_wing.attitude.pid import attitude_controller
from jax_sim.controllers.fixed_wing.rate.pid import rate_controller
from jax_sim.controllers.fixed_wing.speed.pid import speed_controller


@jax.jit
def cascade_pid_step(
    setpoints: jnp.ndarray,
    state: jnp.ndarray,
    pid_state: PIDState,
    config: PIDConfig,
    dt: float,
) -> Tuple[jnp.ndarray, PIDState]:
    """One step of the cascade PID controller.

    Converts high-level attitude/speed commands to actuator outputs.
    This is the main interface between RL agent and physics.

    Args:
        setpoints: [roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd]
                   - roll_cmd: Target roll angle [rad]
                   - pitch_cmd: Target pitch angle [rad]
                   - yaw_rate_cmd: Target yaw rate [rad/s] (direct rate, no outer loop)
                   - speed_cmd: Target airspeed [m/s]
        state: Aircraft state vector [17]:
               [pos(3), vel(3), quat(4), omega(3), actuators(4)]
        pid_state: PIDState with integrals and previous measurements
        config: PIDConfig with all gains and limits
        dt: Timestep [seconds]

    Returns:
        actuators: [aileron, elevator, rudder, throttle]
                   - aileron: [-1, 1] left/right
                   - elevator: [-1, 1] down/up
                   - rudder: [-1, 1] left/right
                   - throttle: [0, 1]
        new_pid_state: Updated PIDState
    """
    # Unpack setpoints
    roll_cmd = setpoints[0]
    pitch_cmd = setpoints[1]
    yaw_rate_cmd = setpoints[2]
    speed_cmd = setpoints[3]

    # Unpack state
    vel = state[3:6]  # Earth-frame velocity
    quat = state[6:10]  # Quaternion [w, x, y, z]
    omega = state[10:13]  # Body rates [p, q, r]

    # 1. Quaternion → Euler angles
    euler = quat_to_euler_jax(quat)
    roll = euler[0]
    pitch = euler[1]
    # yaw = euler[2]  # Not used for control

    # 2. Compute airspeed (magnitude of velocity)
    airspeed = jnp.linalg.norm(vel)

    # 3. Attitude Controller (Outer Loop - P)
    # angle error → rate setpoints
    p_sp, q_sp = attitude_controller(roll_cmd, pitch_cmd, roll, pitch, config)

    # Yaw uses direct rate command (no outer loop for coordinated turn)
    r_sp = jnp.clip(yaw_rate_cmd, -config.rate_limit, config.rate_limit)

    # 4. Rate Controller (Inner Loop - PID)
    rate_sp = jnp.array([p_sp, q_sp, r_sp])
    rates = omega  # [p, q, r] from state

    actuators_rates, pid_state_1 = rate_controller(
        rate_sp, rates, pid_state, config, dt
    )

    # Note: Sign conventions adjusted based on physics model
    # The elevator needs to be negated because in the physics model,
    # positive elevator increases tail lift which pitches the nose DOWN,
    # but we want positive pitch command to pitch UP.
    aileron = actuators_rates[0]
    elevator = -actuators_rates[1]  # Negate for correct pitch direction
    rudder = actuators_rates[2]

    # 5. Speed Controller (Parallel - PI)
    throttle, pid_state_2 = speed_controller(
        speed_cmd, airspeed, pid_state_1, config, dt
    )

    # 6. Pack outputs
    actuators = jnp.array([aileron, elevator, rudder, throttle])

    return actuators, pid_state_2


def init_pid_state() -> PIDState:
    """Initialize a fresh PIDState.

    Convenience function for creating initial controller state.
    """
    return create_pid_state()


@jax.jit
def reset_pid_state(pid_state: PIDState) -> PIDState:
    """Reset PIDState to zeros (preserving structure).

    Call this on episode reset to clear integrators.
    """
    return PIDState(
        rate_integral=jnp.zeros(3),
        speed_integral=0.0,
        prev_rates=jnp.zeros(3),
    )

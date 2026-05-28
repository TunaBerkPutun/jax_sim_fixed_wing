"""Loss function for PID tuning.

Evaluates PID configs by running step-response simulations.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from jax_sim.vehicles.fixed_wing.tier1 import equations_of_motion
from jax_sim.controllers.fixed_wing.pid_gains import PIDConfig, PIDState, create_pid_config
from jax_sim.controllers.fixed_wing.cascade_pid import cascade_pid_step
from jax_sim.controllers.fixed_wing.rate.pid import rate_controller
from jax_sim.utils.quaternion import quat_to_euler_jax


# Parameter indices in the flat array
# [tau_roll, tau_pitch, rate_kp(3), rate_ki(3), rate_kd(3), speed_kp, speed_ki, throttle_ff]
# Total: 2 + 3 + 3 + 3 + 3 = 14 parameters
N_PARAMS = 14

# Parameter bounds [min, max] for each parameter
PARAM_BOUNDS = jnp.array([
    [0.2, 1.0],    # tau_roll
    [0.2, 1.0],    # tau_pitch
    [0.0, 2.0],    # rate_kp[0] (roll)
    [0.0, 2.0],    # rate_kp[1] (pitch)
    [0.0, 1.0],    # rate_kp[2] (yaw)
    [0.0, 20.0],   # rate_ki[0]
    [0.0, 20.0],   # rate_ki[1]
    [0.0, 20.0],   # rate_ki[2]
    [0.0, 0.1],    # rate_kd[0]
    [0.0, 0.1],    # rate_kd[1]
    [0.0, 0.1],    # rate_kd[2]
    [0.1, 1.0],    # speed_kp
    [0.0, 0.3],    # speed_ki
    [0.3, 0.8],    # throttle_ff
])

# Rate-only parameter indices: [rate_kp(3), rate_ki(3), rate_kd(3)]
N_RATE_PARAMS = 9
RATE_PARAM_BOUNDS = jnp.array([
    [0.0, 2.0],    # rate_kp[0] (roll)
    [0.0, 2.0],    # rate_kp[1] (pitch)
    [0.0, 1.0],    # rate_kp[2] (yaw)
    [0.0, 20.0],   # rate_ki[0]
    [0.0, 20.0],   # rate_ki[1]
    [0.0, 20.0],   # rate_ki[2]
    [0.0, 0.1],    # rate_kd[0]
    [0.0, 0.1],    # rate_kd[1]
    [0.0, 0.1],    # rate_kd[2]
])

RATE_SINE_FREQ_HZ = 0.5
RATE_THROTTLE_CMDS = (0.2, 0.9)
RATE_AXES = (0, 1, 2)


def get_param_bounds() -> jnp.ndarray:
    """Return parameter bounds array."""
    return PARAM_BOUNDS


def params_to_config(params: jnp.ndarray) -> PIDConfig:
    """Convert flat parameter array to PIDConfig.

    Args:
        params: Array of shape (14,)

    Returns:
        PIDConfig with the given parameters
    """
    return PIDConfig(
        tau_roll=params[0],
        tau_pitch=params[1],
        rate_kp=params[2:5],
        rate_ki=params[5:8],
        rate_kd=params[8:11],
        speed_kp=params[11],
        speed_ki=params[12],
        throttle_ff=params[13],
        rate_limit=1.047,  # Fixed at 60 deg/s
        integral_limit=2.0,  # Fixed
    )


def config_to_params(config: PIDConfig) -> jnp.ndarray:
    """Convert PIDConfig to flat parameter array.

    Args:
        config: PIDConfig instance

    Returns:
        Array of shape (14,)
    """
    return jnp.concatenate([
        jnp.array([config.tau_roll, config.tau_pitch]),
        config.rate_kp,
        config.rate_ki,
        config.rate_kd,
        jnp.array([config.speed_kp, config.speed_ki, config.throttle_ff]),
    ])


def config_to_rate_params(config: PIDConfig) -> jnp.ndarray:
    """Convert PIDConfig to rate-only parameter array."""
    return jnp.concatenate([
        config.rate_kp,
        config.rate_ki,
        config.rate_kd,
    ])


def get_rate_param_bounds() -> jnp.ndarray:
    """Return rate-only parameter bounds array."""
    return RATE_PARAM_BOUNDS


def rate_params_to_config(params: jnp.ndarray) -> PIDConfig:
    """Convert rate-only params to PIDConfig using defaults for others."""
    base = create_pid_config()
    return PIDConfig(
        tau_roll=base.tau_roll,
        tau_pitch=base.tau_pitch,
        rate_kp=params[0:3],
        rate_ki=params[3:6],
        rate_kd=params[6:9],
        speed_kp=base.speed_kp,
        speed_ki=base.speed_ki,
        throttle_ff=base.throttle_ff,
        rate_limit=base.rate_limit,
        integral_limit=base.integral_limit,
    )


def get_initial_params() -> jnp.ndarray:
    """Return reasonable initial parameter guess."""
    return jnp.array([
        0.5, 0.5,           # tau_roll, tau_pitch
        0.5, 0.5, 0.3,      # rate_kp
        0.1, 0.1, 0.05,     # rate_ki
        0.02, 0.02, 0.01,   # rate_kd
        0.5, 0.1, 0.6,      # speed_kp, speed_ki, throttle_ff
    ])


def get_rate_initial_params() -> jnp.ndarray:
    """Return reasonable initial guess for rate-only tuning."""
    return jnp.array([
        0.5, 0.5, 0.3,      # rate_kp
        0.1, 0.1, 0.05,     # rate_ki
        0.02, 0.02, 0.01,   # rate_kd
    ])


def sample_rate_amplitude(key: jax.Array) -> jnp.ndarray:
    """Sample sinusoid amplitude for rate tuning [rad/s]."""
    min_amp = jnp.deg2rad(10.0)
    max_amp = jnp.deg2rad(45.0)
    return random.uniform(key, (), minval=min_amp, maxval=max_amp)


def get_rate_scenario_keys(key: jax.Array) -> tuple[jax.Array, ...]:
    """Return per-scenario keys for rate-only evaluation."""
    return tuple(random.split(key, len(RATE_THROTTLE_CMDS) * len(RATE_AXES)))


@jax.jit
def evaluate_single_scenario(
    config: PIDConfig,
    roll_cmd: float,
    pitch_cmd: float,
    yaw_rate_cmd: float,
    speed_cmd: float,
    dt: float = 0.004,
    n_steps: int = 2500,
) -> float:
    """Run one test scenario and return loss.

    Args:
        config: PID configuration to test
        roll_cmd: Constant roll command [rad]
        pitch_cmd: Constant pitch command [rad]
        yaw_rate_cmd: Constant yaw rate command [rad/s]
        speed_cmd: Constant speed command [m/s]
        dt: Timestep
        n_steps: Number of steps

    Returns:
        Loss for this scenario
    """

    state0 = jnp.array([
        0.0, 0.0, -100.0,       # Position (NED)
        20.0, 0.0, 0.0,         # Velocity
        1.0, 0.0, 0.0, 0.0,     # Quaternion (level)
        0.0, 0.0, 0.0,          # Angular velocity
        0.0, 0.0, 0.0, 0.5,     # Actuator states
    ])

    pid_state0 = PIDState(
        rate_integral=jnp.zeros(3),
        speed_integral=0.0,
        prev_rates=jnp.zeros(3),
    )

    # Setpoints: constant roll/pitch/yaw/speed commands
    setpoints = jnp.array([roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd])

    def sim_step(carry, _):
        state, pid_state = carry

        # Get current attitude
        quat = state[6:10]
        euler = quat_to_euler_jax(quat)
        roll, pitch = euler[0], euler[1]

        # Get current speed
        vel = state[3:6]
        speed = jnp.linalg.norm(vel)

        # Get current yaw rate
        yaw_rate = state[12]

        # Run PID controller
        actuators, new_pid_state = cascade_pid_step(
            setpoints, state, pid_state, config, dt
        )

        # Run physics
        next_state = equations_of_motion(state, actuators, dt)

        # Compute step loss
        roll_error = (roll_cmd - roll) ** 2
        pitch_error = (pitch_cmd - pitch) ** 2
        speed_norm = jnp.maximum(speed_cmd, 1.0)
        speed_error = ((speed_cmd - speed) / speed_norm) ** 2
        yaw_rate_error = (yaw_rate_cmd - yaw_rate) ** 2

        tracking_loss = (
            roll_error * 3.0
            + pitch_error * 3.0
            + yaw_rate_error * 1.0
            + speed_error * 0.5
        )
        effort_loss = jnp.sum(actuators[:3] ** 2) * 0.01

        # Heavy crash penalty
        crash_penalty = jnp.where(next_state[2] > 0, 10000.0, 0.0)

        # Heavy divergence penalty (attitude or speed way off)
        divergence = jnp.where(
            (jnp.abs(roll) > 1.5) | (jnp.abs(pitch) > 1.5) | (speed < 5.0),
            1000.0, 0.0
        )

        step_loss = tracking_loss + effort_loss + crash_penalty + divergence

        return (next_state, new_pid_state), step_loss

    _, losses = jax.lax.scan(sim_step, (state0, pid_state0), jnp.arange(n_steps))

    total_loss = jnp.sum(losses)
    total_loss = jnp.where(jnp.isnan(total_loss), 1e8, total_loss)

    return total_loss


@jax.jit
def evaluate_single_rate_scenario(
    config: PIDConfig,
    axis: int,
    throttle_cmd: float,
    key: jax.Array,
    dt: float = 0.004,
    n_steps: int = 2500,
) -> float:
    """Run one rate-only scenario with sinusoidal setpoint."""
    amplitude = sample_rate_amplitude(key)
    omega = 2.0 * jnp.pi * RATE_SINE_FREQ_HZ

    state0 = jnp.array([
        0.0, 0.0, -100.0,       # Position (NED)
        20.0, 0.0, 0.0,         # Velocity
        1.0, 0.0, 0.0, 0.0,     # Quaternion (level)
        0.0, 0.0, 0.0,          # Angular velocity
        0.0, 0.0, 0.0, throttle_cmd,  # Actuator states
    ])

    pid_state0 = PIDState(
        rate_integral=jnp.zeros(3),
        speed_integral=0.0,
        prev_rates=jnp.zeros(3),
    )

    def sim_step(carry, step_idx):
        state, pid_state = carry
        t = step_idx * dt

        rate_sp_scalar = amplitude * jnp.sin(omega * t)
        rate_sp = jnp.zeros(3).at[axis].set(rate_sp_scalar)

        rates = state[10:13]
        actuators_rates, new_pid_state = rate_controller(
            rate_sp, rates, pid_state, config, dt
        )

        actuators = jnp.array([
            actuators_rates[0],
            -actuators_rates[1],
            actuators_rates[2],
            throttle_cmd,
        ])

        next_state = equations_of_motion(state, actuators, dt)

        rate_error = rate_sp - rates
        tracking_loss = jnp.sum(rate_error ** 2)
        effort_loss = jnp.sum(actuators_rates ** 2) * 0.01
        saturation_excess = jnp.maximum(0.0, jnp.abs(actuators_rates) - 0.9)
        saturation_loss = jnp.sum(saturation_excess ** 2) * 0.5

        crash_penalty = jnp.where(next_state[2] > 0, 10000.0, 0.0)

        quat = next_state[6:10]
        euler = quat_to_euler_jax(quat)
        roll, pitch = euler[0], euler[1]
        speed = jnp.linalg.norm(next_state[3:6])
        divergence = jnp.where(
            (jnp.abs(roll) > 1.5) | (jnp.abs(pitch) > 1.5) | (speed < 5.0),
            1000.0, 0.0
        )

        step_loss = tracking_loss + effort_loss + saturation_loss + crash_penalty + divergence
        return (next_state, new_pid_state), step_loss

    _, losses = jax.lax.scan(sim_step, (state0, pid_state0), jnp.arange(n_steps))

    total_loss = jnp.sum(losses)
    total_loss = jnp.where(jnp.isnan(total_loss), 1e8, total_loss)
    return total_loss


def simulate_rate_scenario_debug(
    config: PIDConfig,
    axis: int,
    throttle_cmd: float,
    amplitude: float,
    dt: float = 0.004,
    n_steps: int = 2500,
) -> dict:
    """Run rate-only scenario and return trajectories for debugging."""
    omega = 2.0 * jnp.pi * RATE_SINE_FREQ_HZ

    state = jnp.array([
        0.0, 0.0, -100.0,       # Position (NED)
        20.0, 0.0, 0.0,         # Velocity
        1.0, 0.0, 0.0, 0.0,     # Quaternion (level)
        0.0, 0.0, 0.0,          # Angular velocity
        0.0, 0.0, 0.0, throttle_cmd,  # Actuator states
    ])

    pid_state = PIDState(
        rate_integral=jnp.zeros(3),
        speed_integral=0.0,
        prev_rates=jnp.zeros(3),
    )

    states = []
    rate_sps = []
    actuators_log = []
    losses = []

    for step in range(n_steps):
        t = step * dt
        rate_sp_scalar = amplitude * jnp.sin(omega * t)
        rate_sp = jnp.zeros(3).at[axis].set(rate_sp_scalar)

        rates = state[10:13]
        actuators_rates, pid_state = rate_controller(
            rate_sp, rates, pid_state, config, dt
        )

        actuators = jnp.array([
            actuators_rates[0],
            -actuators_rates[1],
            actuators_rates[2],
            throttle_cmd,
        ])

        next_state = equations_of_motion(state, actuators, dt)

        rate_error = rate_sp - rates
        tracking_loss = jnp.sum(rate_error ** 2)
        effort_loss = jnp.sum(actuators_rates ** 2) * 0.01
        crash_penalty = jnp.where(next_state[2] > 0, 10000.0, 0.0)

        quat = next_state[6:10]
        euler = quat_to_euler_jax(quat)
        roll, pitch = euler[0], euler[1]
        speed = jnp.linalg.norm(next_state[3:6])
        divergence = jnp.where(
            (jnp.abs(roll) > 1.5) | (jnp.abs(pitch) > 1.5) | (speed < 5.0),
            1000.0, 0.0
        )

        step_loss = tracking_loss + effort_loss + crash_penalty + divergence

        states.append(state)
        rate_sps.append(rate_sp_scalar)
        actuators_log.append(actuators)
        losses.append(step_loss)

        state = next_state

    states = jnp.stack(states)
    actuators_log = jnp.stack(actuators_log)
    rate_sps = jnp.stack(rate_sps)
    losses = jnp.stack(losses)
    total_loss = jnp.sum(losses)

    return {
        "states": states,
        "rate_sp": rate_sps,
        "actuators": actuators_log,
        "losses": losses,
        "total_loss": total_loss,
    }


@jax.jit
def evaluate_pid_config(
    params: jnp.ndarray,
    key: jax.Array,
) -> float:
    """Evaluate PID config across MULTIPLE scenarios.

    Tests multiple maneuvers to ensure robust gains:
    - Roll, pitch, and yaw-rate steps
    - Combined roll/pitch commands
    - Speed-hold at multiple setpoints

    Returns average loss across all scenarios.
    """
    params = jnp.clip(params, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
    config = params_to_config(params)

    # Test scenarios: (roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd)
    scenarios = [
        (0.0, 0.0, 0.0, 20.0),                       # Level flight
        (0.0, 0.0, 0.0, 15.0),                       # Slow speed hold
        (0.0, 0.0, 0.0, 25.0),                       # Fast speed hold
        (jnp.deg2rad(10.0), 0.0, 0.0, 20.0),         # +10° roll
        (jnp.deg2rad(-10.0), 0.0, 0.0, 20.0),        # -10° roll
        (jnp.deg2rad(30.0), 0.0, 0.0, 20.0),         # +30° roll
        (jnp.deg2rad(-30.0), 0.0, 0.0, 20.0),        # -30° roll
        (0.0, jnp.deg2rad(5.0), 0.0, 20.0),          # +5° pitch
        (0.0, jnp.deg2rad(-5.0), 0.0, 20.0),         # -5° pitch
        (jnp.deg2rad(15.0), jnp.deg2rad(5.0), 0.0, 20.0),   # Banked climb
        (jnp.deg2rad(-15.0), jnp.deg2rad(-5.0), 0.0, 20.0), # Banked descent
        (0.0, 0.0, jnp.deg2rad(10.0), 20.0),         # +10°/s yaw rate
        (0.0, 0.0, jnp.deg2rad(-10.0), 20.0),        # -10°/s yaw rate
    ]

    total_loss = 0.0
    for roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd in scenarios:
        loss = evaluate_single_scenario(
            config, roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd
        )
        total_loss += loss

    # Return average loss
    return total_loss / len(scenarios)


@jax.jit
def evaluate_rate_config(
    params: jnp.ndarray,
    key: jax.Array,
) -> float:
    """Evaluate rate-only PID config across oscillating rate scenarios."""
    params = jnp.clip(params, RATE_PARAM_BOUNDS[:, 0], RATE_PARAM_BOUNDS[:, 1])
    config = rate_params_to_config(params)

    total_loss = 0.0
    keys = get_rate_scenario_keys(key)
    key_idx = 0
    for throttle_cmd in RATE_THROTTLE_CMDS:
        for axis in RATE_AXES:
            loss = evaluate_single_rate_scenario(
                config, axis, throttle_cmd, keys[key_idx]
            )
            total_loss += loss
            key_idx += 1

    return total_loss / (len(RATE_THROTTLE_CMDS) * len(RATE_AXES))


# Vectorized evaluation for parallel population
evaluate_population = jax.jit(jax.vmap(evaluate_pid_config, in_axes=(0, None)))
evaluate_rate_population = jax.jit(jax.vmap(evaluate_rate_config, in_axes=(0, None)))

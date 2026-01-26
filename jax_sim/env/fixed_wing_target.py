"""FixedWingTarget-v1: Pure JAX RL Environment.

6-DOF fixed-wing aircraft with cascade PID control.
Agent commands high-level setpoints to reach target waypoints.
"""

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jax_sim.physics.dynamics import equations_of_motion
from jax_sim.controllers.cascade_pid import cascade_pid_step
from jax_sim.controllers.pid_gains import PIDConfig, PIDState, create_pid_state
from jax_sim.env.domain_randomization import (
    randomize_target,
    randomize_pid_config,
    randomize_physics_params,
)
from jax_sim.env.observation import get_observation
from jax_sim.env.reward import compute_reward
from jax_sim.env.termination import check_termination


# Environment constants
DT = 0.004  # Timestep (250 Hz)
INITIAL_POS = jnp.array([0.0, 0.0, -100.0])  # NED: 100m altitude
INITIAL_VEL = jnp.array([20.0, 0.0, 0.0])  # 20 m/s forward
INITIAL_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])  # Level flight
INITIAL_OMEGA = jnp.zeros(3)  # No rotation
INITIAL_ACTUATORS = jnp.array([0.0, 0.0, 0.0, 0.5])  # Neutral + 50% throttle

# Action scaling (maps [-1, 1] to physical units)
ROLL_SCALE = 0.78  # ±45 degrees
PITCH_SCALE = 0.35  # ±20 degrees
YAW_RATE_SCALE = 0.26  # ±15 deg/s
SPEED_DELTA_SCALE = 10.0  # ±10 m/s


class EnvState(NamedTuple):
    """Environment state (fully observable, immutable).

    All state is explicit - no hidden class variables.
    This enables pure functional programming and vmap.
    """
    plane_state: jnp.ndarray  # (17,) [pos(3), vel(3), quat(4), omega(3), actuators(4)]
    target_pos: jnp.ndarray   # (3,) target position in NED
    target_speed: float        # target airspeed [m/s]
    pid_state: PIDState        # PID controller state (integrals, prev_rates)
    pid_config: PIDConfig      # PID gains (randomized per episode)
    time: float                # elapsed time [seconds]
    last_action: jnp.ndarray   # (4,) previous action for smoothness
    key: PRNGKey               # RNG state for stochasticity


@jax.jit
def reset(key: PRNGKey, base_pid_config: PIDConfig = None) -> Tuple[EnvState, jnp.ndarray]:
    """Reset environment to initial state with domain randomization.

    Args:
        key: JAX PRNG key
        base_pid_config: Base PID configuration to randomize (optional)
                        If None, uses a default config

    Returns:
        state: Initial environment state
        obs: Initial observation (19D)

    Domain randomization:
        - Target: Random position (50-150m) and speed (15-25 m/s)
        - PID gains: ±30% noise around base config
        - Physics params: Mass ±20%, servo lag [0.05, 0.15]s
    """
    key, key_target, key_pid, key_phys = jax.random.split(key, 4)

    # Generate random target
    target_pos, target_speed = randomize_target(key_target)

    # Randomize PID config
    if base_pid_config is None:
        # Default PID config (will be replaced with loaded config)
        base_pid_config = PIDConfig(
            tau_roll=0.5,
            tau_pitch=0.5,
            rate_kp=jnp.array([0.5, 0.5, 0.3]),
            rate_ki=jnp.array([0.1, 0.1, 0.05]),
            rate_kd=jnp.array([0.01, 0.01, 0.005]),
            speed_kp=0.5,
            speed_ki=0.1,
            throttle_ff=0.6,
            rate_limit=1.047,  # 60 deg/s
            integral_limit=2.0,
        )

    pid_config = randomize_pid_config(base_pid_config, key_pid)

    # Randomize physics params (currently not used, but available for future)
    phys_params = randomize_physics_params(key_phys)

    # Initialize aircraft state (fixed initial conditions)
    plane_state = jnp.concatenate([
        INITIAL_POS,
        INITIAL_VEL,
        INITIAL_QUAT,
        INITIAL_OMEGA,
        INITIAL_ACTUATORS,
    ])

    # Initialize PID state (zero integrals)
    pid_state = create_pid_state()

    # Initial action (neutral)
    last_action = jnp.zeros(4)

    # Create environment state
    env_state = EnvState(
        plane_state=plane_state,
        target_pos=target_pos,
        target_speed=target_speed,
        pid_state=pid_state,
        pid_config=pid_config,
        time=0.0,
        last_action=last_action,
        key=key,
    )

    # Get initial observation
    obs = get_obs(env_state)

    return env_state, obs


@jax.jit
def step(
    state: EnvState,
    action: jnp.ndarray,
    key: PRNGKey,
) -> Tuple[EnvState, jnp.ndarray, float, bool, Dict]:
    """Step environment forward one timestep.

    Args:
        state: Current environment state
        action: (4,) action vector in [-1, 1]
                [roll_setpoint, pitch_setpoint, yaw_rate_setpoint, speed_delta]
        key: JAX PRNG key (for future stochasticity, currently unused)

    Returns:
        next_state: Updated environment state
        obs: Observation (19D)
        reward: Scalar reward
        done: Boolean termination flag
        info: Dictionary with debugging/logging info

    Step logic:
        1. Scale action from [-1, 1] to physical units
        2. Run cascade PID (setpoints → actuators)
        3. Step physics (actuators → next_plane_state)
        4. Compute reward
        5. Check termination
        6. Extract observation
    """
    # 1. Scale action from [-1, 1] to physical units
    roll_cmd = action[0] * ROLL_SCALE
    pitch_cmd = action[1] * PITCH_SCALE
    yaw_rate_cmd = action[2] * YAW_RATE_SCALE
    speed_delta = action[3] * SPEED_DELTA_SCALE

    # Target speed is base speed + delta
    speed_cmd = state.target_speed + speed_delta

    # Clamp speed to reasonable range [10, 35] m/s
    speed_cmd = jnp.clip(speed_cmd, 10.0, 35.0)

    # Setpoints for cascade PID
    setpoints = jnp.array([roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd])

    # 2. Run cascade PID controller
    actuators, new_pid_state = cascade_pid_step(
        setpoints, state.plane_state, state.pid_state, state.pid_config, DT
    )

    # 3. Step physics
    next_plane_state = equations_of_motion(state.plane_state, actuators, DT)

    # 4. Compute reward (dense shaping + terminal)
    step_reward, reward_info = compute_reward(
        next_plane_state,
        state.target_pos,
        state.target_speed,
        action,
        state.last_action,
    )

    # 5. Check termination
    done, terminal_reward, term_info = check_termination(
        next_plane_state,
        state.target_pos,
        state.target_speed,
        state.time + DT,
    )

    # Total reward = step reward + terminal reward
    total_reward = step_reward + terminal_reward

    # 6. Create next state
    next_state = EnvState(
        plane_state=next_plane_state,
        target_pos=state.target_pos,  # Target doesn't move
        target_speed=state.target_speed,
        pid_state=new_pid_state,
        pid_config=state.pid_config,  # Config doesn't change during episode
        time=state.time + DT,
        last_action=action,
        key=key,
    )

    # 7. Extract observation
    obs = get_obs(next_state)

    # 8. Combine info dicts
    info = {**reward_info, **term_info}
    info["terminal_reward"] = terminal_reward

    return next_state, obs, total_reward, done, info


@jax.jit
def get_obs(state: EnvState) -> jnp.ndarray:
    """Extract observation from environment state.

    Args:
        state: Environment state

    Returns:
        obs: (19,) normalized observation vector
    """
    return get_observation(
        state.plane_state,
        state.target_pos,
        state.target_speed,
        state.last_action,
    )

"""Environment wrappers and factory functions."""

import json
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jax_sim.env.fixed_wing_target import EnvState, reset, step, get_obs
from jax_sim.controllers.fixed_wing.pid_gains import PIDConfig
from jax_sim.physics.wind import WindConfig, DEFAULT_WIND_CONFIG


def load_tuned_pid_config(config_path: str = "tuned_pid_config.json") -> PIDConfig:
    """Load PID configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        PIDConfig with loaded parameters

    JSON format:
        {
            "tau_roll": 0.37,
            "tau_pitch": 0.77,
            "rate_kp": [0.75, 2.0, 0.15],
            "rate_ki": [0.5, 0.10, 0.0],
            "rate_kd": [0.0, 0.002, 0.001],
            "speed_kp": 0.19,
            "speed_ki": 0.09,
            "throttle_ff": 0.74,
            "rate_limit": 1.047,
            "integral_limit": 2.0
        }
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return PIDConfig(
        tau_roll=float(config_dict["tau_roll"]),
        tau_pitch=float(config_dict["tau_pitch"]),
        rate_kp=jnp.array(config_dict["rate_kp"]),
        rate_ki=jnp.array(config_dict["rate_ki"]),
        rate_kd=jnp.array(config_dict["rate_kd"]),
        speed_kp=float(config_dict["speed_kp"]),
        speed_ki=float(config_dict["speed_ki"]),
        throttle_ff=float(config_dict["throttle_ff"]),
        rate_limit=float(config_dict.get("rate_limit", 1.047)),
        integral_limit=float(config_dict.get("integral_limit", 2.0)),
    )


def make_env(
    config_path: str = "tuned_pid_config.json",
    wind_config: WindConfig = DEFAULT_WIND_CONFIG,
) -> Dict:
    """Factory function to create environment with tuned PID config.

    Args:
        config_path: Path to tuned PID configuration JSON

    Returns:
        Dictionary with:
            - reset_fn: Partial reset function with base PID config
            - step_fn: Step function
            - get_obs_fn: Observation extraction function
            - obs_shape: Observation space shape (19,)
            - action_shape: Action space shape (4,)
            - base_pid_config: Loaded PID configuration

    Usage:
        env = make_env("tuned_pid_config.json")
        key = jax.random.PRNGKey(0)
        state, obs = env["reset_fn"](key)
        state, obs, reward, done, info = env["step_fn"](state, action, key)
    """
    if wind_config is None:
        wind_config = DEFAULT_WIND_CONFIG

    # Load base PID config
    try:
        base_pid_config = load_tuned_pid_config(config_path)
        print(f"✓ Loaded PID config from {config_path}")
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default PID config")
        base_pid_config = None

    # Create partial reset and step functions with fixed configs
    reset_fn = partial(reset, base_pid_config=base_pid_config, wind_config=wind_config)
    step_fn = partial(step, wind_config=wind_config)

    return {
        "reset_fn": reset_fn,
        "step_fn": step_fn,
        "get_obs_fn": get_obs,
        "obs_shape": (19,),
        "action_shape": (4,),
        "base_pid_config": base_pid_config,
        "wind_config": wind_config,
    }


@jax.jit
def auto_reset_step(
    state: EnvState,
    action: jnp.ndarray,
    key: PRNGKey,
    base_pid_config: PIDConfig,
    wind_config: WindConfig = DEFAULT_WIND_CONFIG,
) -> Tuple[EnvState, jnp.ndarray, float, bool, Dict]:
    """Step with automatic reset on episode termination.

    When done=True, automatically resets the environment while preserving
    the terminal observation in info["terminal_observation"].

    Args:
        state: Current environment state
        action: Action to take
        key: PRNG key
        base_pid_config: Base PID config for reset

    Returns:
        next_state: Next state (reset if done)
        obs: Observation (from reset if done, else from step)
        reward: Reward from step
        done: Boolean done flag
        info: Info dict with terminal_observation if done

    Usage:
        # Wrap with functools.partial
        auto_step_fn = partial(auto_reset_step, base_pid_config=config)

        # Or use in training loop
        for _ in range(num_steps):
            state, obs, reward, done, info = auto_reset_step(
                state, action, key, base_pid_config
            )
            # state is automatically reset if done=True
    """
    # Step environment
    key_step, key_reset = jax.random.split(key)
    next_state, obs, reward, done, info = step(state, action, key_step, wind_config)

    # If done, reset (but keep terminal obs in info)
    reset_state, reset_obs = reset(key_reset, base_pid_config, wind_config)

    # Use jax.tree.map to conditionally select reset vs step state
    final_state = jax.tree.map(
        lambda r, n: jnp.where(done, r, n),
        reset_state,
        next_state,
    )

    # Observation: use reset_obs if done, else step obs
    final_obs = jnp.where(done, reset_obs, obs)

    # Store true terminal observation before reset
    info["terminal_observation"] = obs

    return final_state, final_obs, reward, done, info

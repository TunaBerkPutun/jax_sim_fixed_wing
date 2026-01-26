"""RL Environment for Fixed-Wing Aircraft.

Pure JAX (gymnax-style) environment for trajectory tracking with cascade PID control.
"""

from jax_sim.env.fixed_wing_target import (
    EnvState,
    reset,
    step,
    get_obs,
)

__all__ = [
    "EnvState",
    "reset",
    "step",
    "get_obs",
]

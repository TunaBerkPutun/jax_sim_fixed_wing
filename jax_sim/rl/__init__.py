"""PPO Training System for FixedWingTarget-v1.

Uses Flax NNX for neural networks and Optax for optimization.
"""

from jax_sim.rl.config import PPOConfig
from jax_sim.rl.models import Actor, Critic

__all__ = [
    "PPOConfig",
    "Actor",
    "Critic",
]

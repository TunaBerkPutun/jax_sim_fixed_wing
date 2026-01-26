"""Rollout storage and episode statistics.

Immutable dataclasses for storing rollout data and tracking episode metrics.
"""

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class RolloutBuffer:
    """Storage for one rollout (num_steps × num_envs).

    All arrays have shape (num_steps, num_envs, ...).
    Immutable (frozen dataclass) required for JAX JIT compilation.
    """
    obs: jnp.ndarray          # (T, N, 19) observations
    actions: jnp.ndarray      # (T, N, 4) continuous actions
    rewards: jnp.ndarray      # (T, N) rewards
    dones: jnp.ndarray        # (T, N) boolean episode termination
    values: jnp.ndarray       # (T, N) value predictions
    log_probs: jnp.ndarray    # (T, N) action log probabilities

    # Computed after rollout by GAE
    advantages: jnp.ndarray   # (T, N) advantage estimates
    returns: jnp.ndarray      # (T, N) value targets


@flax.struct.dataclass
class EpisodeStats:
    """Running episode statistics for logging.

    Tracks cumulative returns/lengths per environment and stores
    completed episode metrics for reporting.
    """
    episode_returns: jnp.ndarray     # (num_envs,) cumulative returns
    episode_lengths: jnp.ndarray     # (num_envs,) cumulative lengths

    # Completed episodes (for logging)
    completed_returns: jnp.ndarray   # (num_envs,) last completed return
    completed_lengths: jnp.ndarray   # (num_envs,) last completed length

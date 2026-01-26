"""PPO algorithm core using Flax NNX.

Implements action sampling, GAE computation, loss function, and update loop.
"""

from typing import Dict, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from jax_sim.rl.buffers import RolloutBuffer
from jax_sim.rl.config import PPOConfig


def sample_action(
    actor: nnx.Module,
    obs: jnp.ndarray,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample action from Gaussian policy.

    Args:
        actor: Actor module (NNX)
        obs: Observations (batch, 19) or (19,)
        key: JAX PRNG key

    Returns:
        action: Sampled actions (batch, 4) clipped to [-1, 1]
        log_prob: Log probabilities (batch,)
    """
    # Forward pass
    mean, log_std = actor(obs)
    std = jnp.exp(log_std)

    # Sample from N(mean, std) using reparameterization trick
    noise = jax.random.normal(key, mean.shape)
    action_raw = mean + noise * std

    # Clip to action bounds
    action = jnp.clip(action_raw, -1.0, 1.0)

    # Compute log probability (before clipping)
    # log p(a) = -0.5 * [(a-μ)/σ]² - log(σ) - 0.5*log(2π)
    log_prob = -0.5 * jnp.sum(
        ((action_raw - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )

    return action, log_prob


def get_action_and_value(
    actor: nnx.Module,
    critic: nnx.Module,
    obs: jnp.ndarray,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample action and compute value (used during rollout).

    Args:
        actor: Actor module
        critic: Critic module
        obs: Observations (batch, 19)
        key: PRNG key

    Returns:
        action: Sampled actions (batch, 4)
        log_prob: Log probabilities (batch,)
        value: State values (batch,)
    """
    action, log_prob = sample_action(actor, obs, key)
    value = critic(obs)
    return action, log_prob, value


@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    values: jnp.ndarray,
    next_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    Uses jax.lax.scan for efficiency (reverse scan from terminal state).

    Args:
        rewards: (T, N) rewards
        dones: (T, N) episode termination flags
        values: (T, N) value predictions
        next_value: (N,) bootstrap value from next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: (T, N) advantage estimates
        returns: (T, N) value targets (advantages + values)
    """

    def gae_step(carry, inp):
        """Single GAE step (reverse time)."""
        gae = carry
        reward, done, value, next_value = inp

        # TD error: δ = r + γV(s') - V(s)
        delta = reward + gamma * next_value * (1 - done) - value

        # GAE recursion: A_t = δ_t + γλ(1-done) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return gae, gae

    # Bootstrap from next value
    values_ext = jnp.concatenate([values, next_value[None, :]], axis=0)
    dones_ext = jnp.concatenate([dones, jnp.zeros((1, dones.shape[1]))], axis=0)

    # Reverse scan (compute advantages backward in time)
    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros((rewards.shape[1],)),  # Initial GAE = 0
        (rewards, dones_ext[:-1], values, values_ext[1:]),
        reverse=True,
    )

    # Returns = advantages + values (for value loss)
    returns = advantages + values

    return advantages, returns


def ppo_loss(
    actor: nnx.Module,
    critic: nnx.Module,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    config: PPOConfig,
) -> Tuple[float, Dict]:
    """PPO clipped surrogate loss.

    Args:
        actor: Actor module
        critic: Critic module
        obs: Observations (batch, 19)
        actions: Actions taken (batch, 4)
        old_log_probs: Log probs from rollout (batch,)
        advantages: Advantage estimates (batch,)
        returns: Value targets (batch,)
        config: PPO configuration

    Returns:
        loss: Total loss (scalar)
        info: Dictionary with loss components
    """
    # Get current predictions
    mean, log_std = actor(obs)
    std = jnp.exp(log_std)

    # Recompute log probs for actions taken
    log_prob = -0.5 * jnp.sum(
        ((actions - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )

    # Policy loss (PPO clip)
    ratio = jnp.exp(log_prob - old_log_probs)

    # Normalize advantages (optional, usually helps)
    if config.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped surrogate objective
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss (MSE)
    value_pred = critic(obs)
    v_loss = 0.5 * ((value_pred - returns) ** 2).mean()

    # Entropy bonus (from log_std)
    # H(Gaussian) = 0.5 * (log(2πe) + log(σ²)) = 0.5 * (log_std + constant)
    entropy = 0.5 * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

    # Total loss
    loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy

    # Metrics for logging
    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
    clip_frac = (jnp.abs(ratio - 1.0) > config.clip_coef).mean()

    info = {
        "loss/total": loss,
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy,
        "debug/approx_kl": approx_kl,
        "debug/ratio_mean": ratio.mean(),
        "debug/ratio_std": ratio.std(),
        "debug/clip_frac": clip_frac,
    }

    return loss, info


def update_ppo_simple(
    actor: nnx.Module,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
) -> Dict:
    """Simplified PPO update using Python loops (no scan).

    Easier to debug than scan version. Can optimize later if needed.

    Args:
        actor: Actor module (updated in-place)
        critic: Critic module (updated in-place)
        optimizer_actor: Actor optimizer (updates actor)
        optimizer_critic: Critic optimizer (updates critic)
        buffer: Rollout data
        config: PPO config

    Returns:
        metrics: Dictionary with averaged loss/debug metrics
    """
    all_metrics = []

    # Flatten buffer (T, N, ...) → (T*N, ...)
    def flatten(x):
        return x.reshape((-1,) + x.shape[2:])

    flat_buffer = jax.tree.map(flatten, buffer)

    # Multiple epochs over the same data
    for epoch in range(config.update_epochs):
        # Shuffle data
        perm = jax.random.permutation(
            jax.random.PRNGKey(epoch),  # Different shuffle each epoch
            flat_buffer.obs.shape[0],
        )
        shuffled = jax.tree.map(lambda x: x[perm], flat_buffer)

        # Process minibatches
        for i in range(config.num_minibatches):
            start = i * config.minibatch_size
            end = start + config.minibatch_size
            minibatch = jax.tree.map(lambda x: x[start:end], shuffled)

            # Compute loss and gradients
            def loss_fn(actor, critic):
                return ppo_loss(
                    actor,
                    critic,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.log_probs,
                    minibatch.advantages,
                    minibatch.returns,
                    config,
                )

            (loss, metrics), (actor_grads, critic_grads) = nnx.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(actor, critic)

            # Apply gradients (updates modules in-place)
            optimizer_actor.update(actor, actor_grads)
            optimizer_critic.update(critic, critic_grads)

            all_metrics.append(metrics)

    # Average metrics across all epochs and minibatches
    avg_metrics = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)), *all_metrics)

    return avg_metrics

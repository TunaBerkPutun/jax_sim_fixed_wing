"""Main PPO training loop using Flax NNX.

Integrates environment, networks, and PPO algorithm for end-to-end training.
"""

import time
from pathlib import Path
from typing import Dict

from flax import nnx
import jax
import jax.numpy as jnp
import optax

try:
    from tensorboardX import SummaryWriter
except ImportError:
    # Fallback: simple file logging
    class SummaryWriter:
        def __init__(self, log_dir):
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "metrics.txt"

        def add_scalar(self, tag, value, step):
            with open(self.log_file, "a") as f:
                f.write(f"{step},{tag},{value}\n")

        def add_text(self, tag, text):
            pass

        def close(self):
            pass

from jax_sim.env.wrappers import make_env
from jax_sim.rl.models import Actor, Critic
from jax_sim.rl.config import PPOConfig
from jax_sim.rl.buffers import RolloutBuffer, EpisodeStats
from jax_sim.rl.ppo import (
    sample_action,
    get_action_and_value,
    compute_gae,
    update_ppo_simple,
)


def collect_rollout(
    actor: nnx.Module,
    critic: nnx.Module,
    env_states,
    obs: jnp.ndarray,
    episode_stats: EpisodeStats,
    vec_step,
    rngs: nnx.Rngs,
    config: PPOConfig,
):
    """Collect one rollout (num_steps × num_envs).

    Uses simple Python loop (no scan) for easier debugging.

    Args:
        actor: Actor module
        critic: Critic module
        env_states: Current environment states (num_envs,)
        obs: Current observations (num_envs, 19)
        episode_stats: Episode statistics
        vec_step: Vectorized step function
        rngs: NNX RNG stream
        config: PPO config

    Returns:
        buffer: RolloutBuffer with collected data
        env_states: Updated environment states
        obs: Final observations
        episode_stats: Updated episode statistics
    """
    rollout_data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "log_probs": [],
    }

    # Use standard JAX keys for environment stepping
    step_key = jax.random.PRNGKey(config.seed + 1000)

    for step_idx in range(config.num_steps):
        # Sample actions for all environments
        step_key, *action_keys = jax.random.split(step_key, config.num_envs + 1)

        # Vectorize action sampling over batch
        actions, log_probs = jax.vmap(
            lambda o, k: sample_action(actor, o, k)
        )(obs, jnp.array(action_keys))

        # Get values (vectorized over batch)
        values = jax.vmap(critic)(obs)

        # Step all environments in parallel
        step_key, *env_step_keys = jax.random.split(step_key, config.num_envs + 1)
        env_states, next_obs, rewards, dones, infos = vec_step(
            env_states, actions, jnp.array(env_step_keys)
        )

        # Update episode statistics
        episode_stats = update_episode_stats(episode_stats, rewards, dones)

        # Store transition
        rollout_data["obs"].append(obs)
        rollout_data["actions"].append(actions)
        rollout_data["rewards"].append(rewards)
        rollout_data["dones"].append(dones)
        rollout_data["values"].append(values)
        rollout_data["log_probs"].append(log_probs)

        obs = next_obs

    # Stack lists into arrays (T, N, ...)
    buffer = RolloutBuffer(
        obs=jnp.stack(rollout_data["obs"]),
        actions=jnp.stack(rollout_data["actions"]),
        rewards=jnp.stack(rollout_data["rewards"]),
        dones=jnp.stack(rollout_data["dones"]),
        values=jnp.stack(rollout_data["values"]),
        log_probs=jnp.stack(rollout_data["log_probs"]),
        advantages=jnp.zeros_like(jnp.stack(rollout_data["rewards"])),
        returns=jnp.zeros_like(jnp.stack(rollout_data["rewards"])),
    )

    return buffer, env_states, obs, episode_stats


def update_episode_stats(
    stats: EpisodeStats, rewards: jnp.ndarray, dones: jnp.ndarray
) -> EpisodeStats:
    """Update episode statistics (functional).

    Args:
        stats: Current episode stats
        rewards: Step rewards (num_envs,)
        dones: Termination flags (num_envs,)

    Returns:
        Updated episode stats
    """
    new_returns = stats.episode_returns + rewards
    new_lengths = stats.episode_lengths + 1

    # Reset accumulators on episode completion
    episode_returns = new_returns * (1 - dones)
    episode_lengths = new_lengths * (1 - dones)

    # Store completed episode metrics
    completed_returns = jnp.where(dones, new_returns, stats.completed_returns)
    completed_lengths = jnp.where(dones, new_lengths, stats.completed_lengths)

    return EpisodeStats(
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        completed_returns=completed_returns,
        completed_lengths=completed_lengths,
    )


def log_metrics(
    writer: SummaryWriter,
    update: int,
    global_step: int,
    metrics: Dict,
    episode_stats: EpisodeStats,
):
    """Log training metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        update: Current update number
        global_step: Total environment steps
        metrics: Loss/debug metrics from PPO update
        episode_stats: Episode statistics
    """
    # Training metrics
    for key, value in metrics.items():
        writer.add_scalar(key, float(value), global_step)

    # Episode metrics (average over environments)
    avg_return = episode_stats.completed_returns.mean()
    avg_length = episode_stats.completed_lengths.mean()

    writer.add_scalar("charts/avg_episodic_return", float(avg_return), global_step)
    writer.add_scalar("charts/avg_episodic_length", float(avg_length), global_step)

    # Console output
    print(
        f"Update {update:4d} | Step {global_step:8d} | "
        f"Return {avg_return:7.1f} | Length {avg_length:5.0f} | "
        f"Loss {metrics['loss/total']:6.3f}"
    )


def train(config: PPOConfig, run_name: str):
    """Main PPO training loop.

    Args:
        config: PPO configuration
        run_name: Name for this run (used for logging/checkpointing)

    Returns:
        actor: Trained actor module
        critic: Trained critic module
    """
    print("\n" + "=" * 70)
    print(f"PPO Training: {run_name}")
    print("=" * 70)
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Num updates: {config.num_updates}")
    print(f"Batch size: {config.batch_size} ({config.num_envs} envs × {config.num_steps} steps)")
    print("=" * 70 + "\n")

    # Setup logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(config).items()]),
    )

    # Setup RNG
    rngs = nnx.Rngs(config.seed)
    base_key = jax.random.PRNGKey(config.seed)

    # Environment setup
    print("Loading environment...")
    env_dict = make_env("tuned_pid_config.json")

    # Vectorize environment functions
    vec_reset = jax.vmap(env_dict["reset_fn"])
    vec_step = jax.vmap(env_dict["step_fn"], in_axes=(0, 0, 0))

    # Initialize networks
    print("Initializing networks...")
    actor = Actor(
        obs_dim=19,
        action_dim=4,
        hidden_sizes=config.actor_hidden,
        rngs=rngs,
    )

    critic = Critic(
        obs_dim=19,
        hidden_sizes=config.critic_hidden,
        rngs=rngs,
    )

    # Print network info
    actor_params = nnx.state(actor, nnx.Param)
    critic_params = nnx.state(critic, nnx.Param)
    print(f"  Actor params: {len(jax.tree.leaves(actor_params))}")
    print(f"  Critic params: {len(jax.tree.leaves(critic_params))}")

    # Setup optimizers
    if config.anneal_lr:
        schedule = optax.linear_schedule(
            config.learning_rate,
            0.0,
            config.num_updates * config.update_epochs * config.num_minibatches,
        )
    else:
        schedule = config.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(schedule, eps=1e-5),
    )

    optimizer_actor = nnx.Optimizer(actor, tx, wrt=nnx.Param)
    optimizer_critic = nnx.Optimizer(critic, tx, wrt=nnx.Param)

    # Initialize environments
    print(f"Initializing {config.num_envs} environments...")
    base_key, *env_keys = jax.random.split(base_key, config.num_envs + 1)
    env_states, obs = vec_reset(jnp.array(env_keys))

    episode_stats = EpisodeStats(
        episode_returns=jnp.zeros(config.num_envs),
        episode_lengths=jnp.zeros(config.num_envs),
        completed_returns=jnp.zeros(config.num_envs),
        completed_lengths=jnp.zeros(config.num_envs),
    )

    print("\nStarting training...\n")

    # Training loop
    global_step = 0
    start_time = time.time()

    for update in range(config.num_updates):
        update_start = time.time()

        # Collect rollout
        buffer, env_states, obs, episode_stats = collect_rollout(
            actor,
            critic,
            env_states,
            obs,
            episode_stats,
            vec_step,
            rngs,
            config,
        )

        # Compute GAE
        next_value = jax.vmap(critic)(obs)
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values,
            next_value,
            config.gamma,
            config.gae_lambda,
        )

        buffer = buffer.replace(advantages=advantages, returns=returns)

        # Update policy (optimizers update actor/critic in-place)
        metrics = update_ppo_simple(
            actor,
            critic,
            optimizer_actor,
            optimizer_critic,
            buffer,
            config,
        )

        global_step += config.batch_size

        # Log metrics
        if update % config.log_interval == 0:
            log_metrics(writer, update, global_step, metrics, episode_stats)

            # SPS (steps per second)
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            writer.add_scalar("charts/SPS", sps, global_step)

        # Save checkpoint
        if update % config.save_interval == 0 and update > 0:
            from jax_sim.rl.checkpoint import save_checkpoint

            save_checkpoint(
                actor, critic, optimizer_actor, optimizer_critic, update, run_name
            )

        # Update timing
        update_time = time.time() - update_start
        writer.add_scalar("charts/update_time", update_time, global_step)

    # Final metrics
    elapsed_total = time.time() - start_time
    final_sps = config.total_timesteps / elapsed_total

    print("\n" + "=" * 70)
    print(f"Training Complete!")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Average SPS: {final_sps:,.0f}")
    print(f"  Final return: {episode_stats.completed_returns.mean():.1f}")
    print("=" * 70 + "\n")

    writer.close()

    return actor, critic

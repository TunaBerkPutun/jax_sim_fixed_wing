#!/usr/bin/env python3
"""PPO training script for FixedWingTarget-v1.

Uses Flax NNX for models and Optax for optimization.

Usage:
    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --num-envs 32 --learning-rate 1e-4
    uv run python scripts/train_ppo.py --total-timesteps 5000000
"""

import time

import tyro

from jax_sim.rl.config import PPOConfig
from jax_sim.rl.train import train


def main():
    """Parse config and start training."""
    # Parse config from command line
    config = tyro.cli(PPOConfig)

    # Generate run name
    run_name = f"ppo_{config.env_id}_s{config.seed}_{int(time.time())}"

    print("\n" + "=" * 70)
    print(f"Starting PPO Training")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Config:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")

    # Train
    actor, critic = train(config, run_name)

    print("\n✅ Training complete!")
    print(f"Logs saved to: runs/{run_name}")
    print(f"Checkpoints saved to: checkpoints/{run_name}\n")


if __name__ == "__main__":
    main()

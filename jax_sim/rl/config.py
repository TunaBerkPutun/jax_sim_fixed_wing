"""PPO configuration for continuous control."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO hyperparameters optimized for continuous control.

    Based on CleanRL best practices for continuous action spaces.
    """

    # Environment
    env_id: str = "FixedWingTarget-v1"
    num_envs: int = 16  # Vectorized environments (balance diversity vs memory)

    # Training
    total_timesteps: int = 2_000_000  # 2M steps
    num_steps: int = 256  # Rollout length per env (long for temporal credit)

    # PPO Algorithm
    learning_rate: float = 3e-4  # Standard for continuous control
    anneal_lr: bool = True  # Linear LR decay to 0
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    num_minibatches: int = 8  # Batch split (4096 / 8 = 512 per minibatch)
    update_epochs: int = 10  # More than Atari (continuous is easier to optimize)
    norm_adv: bool = True  # Normalize advantages
    clip_coef: float = 0.2  # PPO clip epsilon
    clip_vloss: bool = True  # Clip value loss
    ent_coef: float = 0.0  # Entropy bonus (Gaussian has natural exploration)
    vf_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    target_kl: float | None = None  # Early stopping (optional)

    # Network Architecture
    actor_hidden: tuple = (256, 256)  # Actor MLP hidden sizes
    critic_hidden: tuple = (256, 256)  # Critic MLP hidden sizes

    # Logging
    log_interval: int = 10  # Log every N updates
    save_interval: int = 100  # Save checkpoint every N updates
    use_wandb: bool = False  # Enable Weights & Biases logging

    # System
    seed: int = 42  # Random seed

    # Computed fields (set in __post_init__)
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0

    def __post_init__(self):
        """Compute derived hyperparameters."""
        self.batch_size = self.num_envs * self.num_steps  # 16 * 256 = 4096
        self.minibatch_size = self.batch_size // self.num_minibatches  # 512
        self.num_updates = self.total_timesteps // self.batch_size  # ~488 updates

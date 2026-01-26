"""Actor-Critic networks using Flax NNX.

Implements Gaussian policy (Actor) and value function (Critic) for continuous control.
"""

from flax import nnx
import jax.numpy as jnp
import numpy as np


class Actor(nnx.Module):
    """Gaussian policy for continuous actions (Flax NNX).

    Outputs mean and log_std for diagonal Gaussian distribution.
    Uses state-independent log_std (learnable parameter, not network output).
    """

    def __init__(
        self,
        obs_dim: int = 19,
        action_dim: int = 4,
        hidden_sizes: tuple = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Actor network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Tuple of hidden layer sizes
            rngs: NNX RNG stream for initialization
        """
        # Build MLP backbone using Sequential
        layers = []
        dims = [obs_dim] + list(hidden_sizes)

        for i in range(len(hidden_sizes)):
            layers.append(
                nnx.Linear(
                    dims[i],
                    dims[i + 1],
                    kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
            )
            layers.append(nnx.tanh)  # Activation function

        # Use Sequential to wrap layers (NNX-compatible)
        self.backbone = nnx.Sequential(*layers)

        # Mean output head (small init for stability)
        self.mean_head = nnx.Linear(
            hidden_sizes[-1],
            action_dim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        # Log std as learnable parameter (NOT state-dependent)
        # Init to 0.0 → std = exp(0) = 1.0 (reasonable exploration)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Observations (batch, obs_dim)

        Returns:
            mean: Action means (batch, action_dim)
            log_std: Log standard deviations (action_dim,) - broadcastable
        """
        # MLP backbone (Sequential handles forward pass)
        features = self.backbone(x)

        # Mean output
        mean = self.mean_head(features)

        # Log std (parameter, not dependent on state)
        log_std = self.log_std.value

        return mean, log_std


class Critic(nnx.Module):
    """State-value function V(s) (Flax NNX).

    Estimates expected return from a given state.
    """

    def __init__(
        self,
        obs_dim: int = 19,
        hidden_sizes: tuple = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Critic network.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: Tuple of hidden layer sizes
            rngs: NNX RNG stream for initialization
        """
        # Build MLP backbone using Sequential
        layers = []
        dims = [obs_dim] + list(hidden_sizes)

        for i in range(len(hidden_sizes)):
            layers.append(
                nnx.Linear(
                    dims[i],
                    dims[i + 1],
                    kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
            )
            layers.append(nnx.tanh)  # Activation function

        # Use Sequential to wrap layers (NNX-compatible)
        self.backbone = nnx.Sequential(*layers)

        # Scalar value output
        self.value_head = nnx.Linear(
            hidden_sizes[-1],
            1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Observations (batch, obs_dim)

        Returns:
            value: State values (batch,)
        """
        # MLP backbone (Sequential handles forward pass)
        features = self.backbone(x)

        # Value output (squeeze to scalar per sample)
        value = self.value_head(features)
        return value.squeeze(-1)


def create_agent(config, seed: int = 0):
    """Factory function to create Actor and Critic networks.

    Args:
        config: PPOConfig with network architecture settings
        seed: Random seed for initialization

    Returns:
        actor: Actor module
        critic: Critic module
    """
    rngs = nnx.Rngs(seed)

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

    return actor, critic

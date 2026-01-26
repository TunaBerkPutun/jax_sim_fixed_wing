"""Checkpointing utilities for NNX models.

Uses Flax serialization for saving/loading NNX module state.
"""

import pickle
from pathlib import Path

from flax import nnx


def save_checkpoint(
    actor: nnx.Module,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    update: int,
    run_name: str,
):
    """Save checkpoint with NNX state management.

    Args:
        actor: Actor module
        critic: Critic module
        optimizer_actor: Actor optimizer
        optimizer_critic: Critic optimizer
        update: Current update number
        run_name: Run identifier for checkpoint directory
    """
    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Extract state from NNX modules
    _, actor_state = nnx.split(actor)
    _, critic_state = nnx.split(critic)
    _, opt_actor_state = nnx.split(optimizer_actor)
    _, opt_critic_state = nnx.split(optimizer_critic)

    checkpoint = {
        "actor_state": actor_state,
        "critic_state": critic_state,
        "opt_actor_state": opt_actor_state,
        "opt_critic_state": opt_critic_state,
        "update": update,
    }

    # Save with pickle (simple and reliable)
    ckpt_path = ckpt_dir / f"checkpoint_{update}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"✓ Saved checkpoint at update {update} → {ckpt_path}")


def load_checkpoint(
    actor: nnx.Module,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    run_name: str,
    update: int = None,
):
    """Load checkpoint and restore NNX modules.

    Args:
        actor: Actor module (will be updated in-place)
        critic: Critic module (will be updated in-place)
        optimizer_actor: Actor optimizer (will be updated in-place)
        optimizer_critic: Critic optimizer (will be updated in-place)
        run_name: Run identifier
        update: Specific update to load. If None, loads latest.

    Returns:
        update: Loaded update number
    """
    ckpt_dir = Path(f"checkpoints/{run_name}")

    if update is None:
        # Find latest checkpoint
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = checkpoints[-1]
        update = int(ckpt_path.stem.split("_")[-1])
    else:
        ckpt_path = ckpt_dir / f"checkpoint_{update}.pkl"

    # Load checkpoint
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Restore state to modules (updates in-place)
    nnx.update(actor, checkpoint["actor_state"])
    nnx.update(critic, checkpoint["critic_state"])
    nnx.update(optimizer_actor, checkpoint["opt_actor_state"])
    nnx.update(optimizer_critic, checkpoint["opt_critic_state"])

    print(f"✓ Loaded checkpoint from update {update}")
    return update

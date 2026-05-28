"""Slew-rate limiter — port of PX4 `SlewRate::update`.

PX4 reference: `src/lib/slew_rate/SlewRate.hpp:32-55`.

Differentiable everywhere except at the symmetric clip boundary, where the
gradient drops to zero. That's the same behaviour as `jnp.clip`, which is
fine for control synthesis (we don't differentiate *through* a saturated
actuator) and matches PX4's saturate-and-forget semantics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def slew(
    prev: jnp.ndarray,
    target: jnp.ndarray,
    max_rate: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Rate-limit a per-element change toward `target`.

    Args:
        prev: Previous value(s).
        target: Commanded new value(s).
        max_rate: Maximum |Δ/dt| per element (positive, same shape as prev/target).
        dt: Timestep [s].

    Returns:
        new = prev + clip(target - prev, ±max_rate · dt).
    """
    delta = jnp.clip(target - prev, -max_rate * dt, max_rate * dt)
    return prev + delta

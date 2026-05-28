"""High-level simulation APIs with explicit batching."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing.params import AircraftParams
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_AIRCRAFT
from jax_sim.vehicles.fixed_wing.tier1 import equations_of_motion


@jax.jit
def simulate(
    state0: jnp.ndarray,
    commands: jnp.ndarray,
    dt: float = 0.004,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
) -> jnp.ndarray:
    """Simulate a single aircraft over a command sequence.

    Args:
        state0: Initial state, shape (17,)
        commands: Command sequence, shape (steps, 4)
        dt: Timestep [s]
        aircraft: Aircraft configuration

    Returns:
        State trajectory, shape (steps + 1, 17)
    """
    def step_fn(state, cmd):
        next_state = equations_of_motion(state, cmd, dt=dt, aircraft=aircraft)
        return next_state, next_state

    _, states = jax.lax.scan(step_fn, state0, commands)
    return jnp.concatenate([state0[None, :], states], axis=0)


@jax.jit
def simulate_batch(
    state0: jnp.ndarray,
    commands: jnp.ndarray,
    dt: float = 0.004,
    aircraft: AircraftParams = DEFAULT_AIRCRAFT,
) -> jnp.ndarray:
    """Simulate a batch of aircraft over a shared command sequence.

    Args:
        state0: Initial states, shape (batch, 17)
        commands: Command sequence, shape (steps, batch, 4)
        dt: Timestep [s]
        aircraft: Aircraft configuration (shared across batch)

    Returns:
        State trajectories, shape (steps + 1, batch, 17)
    """
    def step_fn(states, cmds_t):
        next_states = jax.vmap(
            equations_of_motion,
            in_axes=(0, 0, None, None),
        )(states, cmds_t, dt, aircraft)
        return next_states, next_states

    _, states = jax.lax.scan(step_fn, state0, commands)
    return jnp.concatenate([state0[None, ...], states], axis=0)

"""Controller interface conventions.

This module documents the shape any controller in `jax_sim.controllers` should
expose so the env (and future PX4 bridge / RL trainer) can swap them without
bespoke wiring. The convention is duck-typed — JAX prefers function passing
over OOP and `jit` doesn't trace through `ABC`s gracefully.

Every controller exposes two callables:

    init_state(initial_plant_state, target, config, **kwargs) -> ControllerState
    step(plant_state, target, ctrl_state, config, dt, wind_ned)
        -> (actuators, new_ctrl_state, debug)

Where:
    plant_state : jnp.ndarray (17,)   sim state [pos(3), vel(3), quat(4), omega(3), actuators(4)]
    target      : controller-specific. Single waypoint, track segment, trajectory, ...
    ctrl_state  : NamedTuple / Flax struct (JAX pytree). All integrators / filters live here.
    config      : NamedTuple / Flax struct (JAX pytree). Fixed gains and limits.
    dt          : float
    wind_ned    : jnp.ndarray (3,)    Oracle NED wind. Pass jnp.zeros(3) if not modelling wind.
    actuators   : jnp.ndarray (4,)    [ail, ele, rud, thr]. ail/ele/rud in [-1,1]; thr in [0,1].
    debug       : NamedTuple. Intermediate setpoints for plotting / imitation.

Both functions must be pure (no Python-level branches on tracer values) so the
whole cascade composes under `jax.jit`, `jax.vmap`, and `jax.grad`.

Reference implementations
-------------------------
- `jax_sim.controllers.fixed_wing.cascade_pid` — simple cascade PID baseline (legacy).
- `jax_sim.controllers.fixed_wing.expert` — PX4-equivalent native JAX cascade.

Future LQR / MPC / learned controllers should match the same signature.
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class ControllerInterface(Protocol):
    """Duck-typed controller protocol.

    Not enforced — used for `isinstance(...)` checks and editor hints only.
    Concrete controllers expose module-level functions, not class methods.
    """

    def init_state(
        self,
        initial_plant_state: jnp.ndarray,
        target: Any,
        config: Any,
        **kwargs: Any,
    ) -> Any: ...

    def step(
        self,
        plant_state: jnp.ndarray,
        target: Any,
        ctrl_state: Any,
        config: Any,
        dt: float,
        wind_ned: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any, Any]: ...


# Lightweight registry. Optional convenience for `env/wrappers.py` so callers
# can pick a controller by string. Use is opt-in; nothing forces registration.
_REGISTRY: dict[str, Tuple[Any, Any]] = {}


def register_controller(name: str, init_fn: Any, step_fn: Any) -> None:
    """Register (init_state, step) callables under a name.

    Args:
        name: Controller identifier (e.g. "simple_pid", "expert", "lqr").
        init_fn: Callable producing the controller state.
        step_fn: Callable producing (actuators, new_state, debug).
    """
    _REGISTRY[name] = (init_fn, step_fn)


def get_controller(name: str) -> Tuple[Any, Any]:
    """Look up a registered controller. Raises KeyError if absent."""
    return _REGISTRY[name]


def list_controllers() -> list[str]:
    """Return the names of all registered controllers."""
    return sorted(_REGISTRY)

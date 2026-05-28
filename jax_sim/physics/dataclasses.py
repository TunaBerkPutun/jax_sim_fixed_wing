"""Vehicle-agnostic parameter pytrees used by the rigid-body integrator.

`EnvironmentParams` and `MassProps` describe scalar properties any rigid body
has: gravity/density in the surrounding fluid, and mass + inertia tensor. They
live here (not in `vehicles/<type>/params.py`) so `physics/rigid_body.py` can
import them without creating a cross-package cycle.

A specific vehicle's `Params` (e.g. `FixedWingParams`) composes these as fields.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class EnvironmentParams:
    """Environment parameters (gravity, fluid density)."""
    gravity: float
    rho: float


@flax.struct.dataclass
class MassProps:
    """Rigid-body mass properties (mass + inertia tensor and its inverse)."""
    mass: float
    inertia: jnp.ndarray
    inertia_inv: jnp.ndarray


__all__ = ["EnvironmentParams", "MassProps"]

"""Attitude controller submodule (outer loop)."""

from jax_sim.controllers.fixed_wing.attitude.pid import (
    attitude_controller,
    attitude_controller_vectorized,
)

__all__ = ["attitude_controller", "attitude_controller_vectorized"]

"""Airspeed-based rate-loop gain scaling.

Port of `FixedwingRateControl::get_airspeed_constrained_scale` (concept) from
PX4 `src/modules/fw_rate_control/FixedwingRateControl.cpp:343-385`.

Idea: control-surface effectiveness scales with dynamic pressure (~ V²). To
keep the rate-loop closed-loop bandwidth roughly invariant over the flight
envelope, the rate-loop output is multiplied by `scale²` where
`scale = V_trim / V` (clipped). FF gains are scaled by `1/scale` so they
remain proportional to the actual rate command in rad/s.

The single function here returns the *unsquared* scale; callers square it
when applying to torque (so the multiplier follows PX4 convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def airspeed_scale(
    airspeed: jnp.ndarray,
    trim: float,
    lo: float,
    hi: float,
    eps: float = 0.1,
) -> jnp.ndarray:
    """Compute the airspeed-based gain-scale.

    Args:
        airspeed: Current TAS [m/s].
        trim: Trim airspeed [m/s] (FW_AIRSPD_TRIM).
        lo: Lower bound on scale (PX4 default ~0.5).
        hi: Upper bound on scale (PX4 default ~2.0).
        eps: Floor on airspeed to avoid division by zero.

    Returns:
        scale = clip(trim / max(airspeed, eps), lo, hi).
    """
    safe_airspeed = jnp.maximum(airspeed, eps)
    return jnp.clip(trim / safe_airspeed, lo, hi)

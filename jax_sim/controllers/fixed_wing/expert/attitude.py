"""Attitude loop — port of PX4 fixed-wing attitude controller.

PX4 reference: `src/modules/fw_att_control/FixedwingAttitudeControl.cpp:301-323`.

Structure (per axis):
    p_sp = (roll_sp  - roll ) / tau_roll        # outer P loop, roll
    q_sp = (pitch_sp - pitch) / tau_pitch + ff_scale · pitchrate_ff
    r_sp = ff_scale · yawrate_ff

Coordinated-turn feed-forward (level coordinated turn at airspeed V):
    yawrate_ff   = g · sin(roll) · cos(pitch) / V
    pitchrate_ff = sin(roll) · yawrate_ff / (cos(roll) · cos(pitch))

These are the body-frame rates required to hold altitude during a banked turn
without sideslip. PX4 derives them from quaternion components directly
(rate_control.cpp:309-311); the euler form here is mathematically identical
and easier to read.

Tilt-gate (PX4:314-316): FF is fully active at |tilt|≤70° and linearly fades to
zero at 75°. `tilt` is the angle between body-z and earth-z, i.e. how far the
aircraft is from upright. Implemented as a smooth interpolation so gradients
stay finite.

Rate-limit clamp at the output, per PX4:321-323.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.expert.types import ExpertConfig
from jax_sim.physics.constants import G

_AIRSPEED_FLOOR = 0.1  # PX4 FixedwingAttitudeControl.cpp:308 — `max(V, 0.1)`


def _smooth_interpolate(x: jnp.ndarray, x0: float, x1: float,
                        y0: float, y1: float) -> jnp.ndarray:
    """Linear interpolation between (x0, y0) and (x1, y1), saturating outside.

    Mirrors PX4 `math::interpolate(x, x0, x1, y0, y1)` semantics. All `jnp.where`
    + `clip` so it's differentiable except at the saturation knees.
    """
    t = jnp.clip((x - x0) / (x1 - x0 + 1e-9), 0.0, 1.0)
    return y0 + t * (y1 - y0)


@jax.jit
def attitude_loop(
    roll_sp: jnp.ndarray,           # commanded roll [rad]
    pitch_sp: jnp.ndarray,          # commanded pitch [rad]
    roll: jnp.ndarray,              # measured roll [rad]
    pitch: jnp.ndarray,             # measured pitch [rad]
    airspeed: jnp.ndarray,          # TAS [m/s]
    config: ExpertConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One step of the PX4-equivalent attitude loop.

    Returns:
        rate_sp: (3,) body rate setpoints [p_sp, q_sp, r_sp] [rad/s].
        tilt: scalar tilt angle [rad] (returned for plotting/debug).
    """
    # --- Outer P -----------------------------------------------------------
    p_sp_p = (roll_sp - roll) / config.tau_roll
    q_sp_p = (pitch_sp - pitch) / config.tau_pitch

    # --- Coordinated-turn FF (PX4:309-311 in euler form) -------------------
    V = jnp.maximum(airspeed, _AIRSPEED_FLOOR)
    s_roll, c_roll = jnp.sin(roll), jnp.cos(roll)
    s_pitch, c_pitch = jnp.sin(pitch), jnp.cos(pitch)

    # q1 = sin(roll) · cos(pitch) — used as the projection of body-z onto earth-y
    q1 = s_roll * c_pitch
    yawrate_ff = G * q1 / V

    # cos_tilt = (3,3) element of body→earth rotation = cos(roll) · cos(pitch).
    cos_tilt = c_roll * c_pitch
    # Pitchrate FF: q1 · yawrate_ff / cos_tilt  (matches PX4:311 in quaternion form).
    pitchrate_ff = q1 * yawrate_ff / jnp.maximum(cos_tilt, 1e-6)

    # --- Tilt gate (smooth fade between 70° and 75°) -----------------------
    # Use `1 - cos_tilt` as the gate variable rather than `arccos(cos_tilt)`.
    # arccos has an unbounded derivative at cos=1 (the level-flight initial
    # condition), which produces NaN gradients during backprop through scans
    # of the cascade. (1-cos_tilt) is smooth and monotonic in tilt, and the
    # thresholds map cleanly: lo↦1-cos(70°)≈0.658, hi↦1-cos(75°)≈0.741.
    gate_input = 1.0 - cos_tilt
    gate_lo = 1.0 - jnp.cos(config.coord_turn_gate_lo)
    gate_hi = 1.0 - jnp.cos(config.coord_turn_gate_hi)
    ff_scale = _smooth_interpolate(gate_input, gate_lo, gate_hi, 1.0, 0.0)

    # `tilt` is only returned for plotting / debug; compute it the *safe* way
    # via the magnitude of body-z (no singular derivative at the level point).
    tilt = jnp.arctan2(jnp.sqrt(jnp.maximum(1.0 - cos_tilt ** 2, 0.0)), cos_tilt)

    p_sp = p_sp_p
    q_sp = q_sp_p + ff_scale * pitchrate_ff
    r_sp = ff_scale * yawrate_ff

    # --- Per-axis rate limits (PX4:321-323) -------------------------------
    rate_sp = jnp.array([p_sp, q_sp, r_sp])
    rate_sp = jnp.clip(rate_sp, -config.rate_limit, config.rate_limit)

    return rate_sp, tilt

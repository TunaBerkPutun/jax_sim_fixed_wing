"""NPFG-lite — Nonlinear Path-Following Guidance for straight segments.

PX4 reference: `src/lib/npfg/DirectionalGuidance.{hpp,cpp}`. The full PX4 NPFG
handles arbitrary curves, wind-aware feasibility detection, and bearing-bound
adaptation. For an "expert" baseline targeting waypoint-to-waypoint flight,
straight segments are sufficient — the missing pieces are noted at the bottom
of this file.

Math (straight segment from `track_start` to `track_end`):
    tangent      = normalize(end - start)
    bearing_path = atan2(tangent.E, tangent.N)                 # rad, North=0
    e_cross      = tangent × (pos - start)  (2-D signed cross product)
                   positive when aircraft is *right* of the path tangent
                   (in NE frame: tangent×East = +1, so East of a Northbound
                    path → positive cross-track error → steer West to return)
    L            = NPFG_PERIOD · V_g / (2π · damping)          # lookahead distance
    course_corr  = atan(e_cross / L)                           # steer toward path
    course_sp    = bearing_path - course_corr

The lookahead `L` scales with ground speed so the cross-track damping ratio
stays constant across the flight envelope (this is NPFG's key idea over L1).

Lateral-acceleration FF for the attitude loop:
    a_lat = V_g² · course_rate  (roughly; exact NPFG form is more elaborate).
    For the straight-segment case where the path doesn't curve, the FF is
    just zero — all lateral accel comes from the course-error closed loop.
    We return zero here; the attitude layer's coordinated-turn FF handles
    bank-to-yaw coupling.

Out of scope for "lite":
- Curved paths (NPFG handles them via path curvature term).
- Wind-induced bearing infeasibility detection (CourseToAirspeedRefMapper
  already bumps airspeed_ref to keep the wind triangle solvable).
- Min-ground-speed enforcement.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.expert.types import ExpertConfig

_GROUND_SPEED_FLOOR = 1.0   # m/s — avoids L collapsing to 0 at zero ground speed
_TWO_PI = 2.0 * jnp.pi


@jax.jit
def npfg_segment_step(
    pos_ne: jnp.ndarray,             # (2,) aircraft position [N, E] [m]
    ground_vel_ne: jnp.ndarray,      # (2,) ground velocity [N, E] [m/s]
    track_start_ne: jnp.ndarray,     # (2,) segment start [N, E] [m]
    track_end_ne: jnp.ndarray,       # (2,) segment end [N, E] [m]
    config: ExpertConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Straight-segment NPFG guidance.

    Returns:
        course_sp: Commanded ground-track direction [rad, North=0].
        lateral_accel_ff: Lateral-acceleration feed-forward [m/s²] (= 0 for
            straight segments).
        cross_track_error: Signed perpendicular distance to the path [m]
            (positive = aircraft is left of the path tangent direction).
    """
    # --- Path geometry ----------------------------------------------------
    delta = track_end_ne - track_start_ne
    # Defensive: avoid normalizing a zero-length segment.
    seg_len = jnp.maximum(jnp.linalg.norm(delta), 1e-6)
    tangent = delta / seg_len  # (N, E)
    # bearing_path = atan2(East, North) — aviation/aerospace convention.
    bearing_path = jnp.arctan2(tangent[1], tangent[0])

    # --- Cross-track error ------------------------------------------------
    rel = pos_ne - track_start_ne
    # 2-D signed cross product: tangent × rel (z-component).
    # tangent = (n, e), rel = (rn, re) → cross_z = n·re - e·rn
    # Sign convention: positive when aircraft is RIGHT of the path tangent.
    # E.g. for a Northbound path (tangent=(1,0)), an aircraft displaced East
    # (rel=(0,+)) gives cross_z = +, which the law below corrects by steering
    # course_sp West (course_corr > 0 → course_sp = bearing_path - course_corr).
    e_cross = tangent[0] * rel[1] - tangent[1] * rel[0]

    # --- Lookahead distance scaled by ground speed ------------------------
    ground_speed = jnp.maximum(jnp.linalg.norm(ground_vel_ne), _GROUND_SPEED_FLOOR)
    # L = period · V_g / (2π · damping). Higher period or lower damping → softer
    # tracking; matches PX4 NPFG's adaptive law for straight segments.
    L = config.npfg_period * ground_speed / (_TWO_PI * config.npfg_damping)

    # --- Course correction ------------------------------------------------
    # atan(e_cross / L) is smooth and bounded to ±π/2. Subtracting from the
    # path bearing steers the aircraft back to the line.
    course_corr = jnp.arctan2(e_cross, L)
    course_sp = bearing_path - course_corr

    # Wrap to (-π, π] for downstream consumers.
    course_sp = jnp.arctan2(jnp.sin(course_sp), jnp.cos(course_sp))

    return course_sp, jnp.asarray(0.0), e_cross

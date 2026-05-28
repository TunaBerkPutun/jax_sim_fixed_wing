"""Wind-triangle solver — port of PX4 `CourseToAirspeedRefMapper`.

PX4 reference: `src/lib/npfg/CourseToAirspeedRefMapper.{hpp,cpp}`.

Geometry — operating in the North-East horizontal plane only (altitude is
handled by TECS):

    V_g  = V_a + W
    course (heading-over-ground) = atan2(V_g.E, V_g.N)
    heading (aircraft yaw)       = atan2(V_a.E, V_a.N)

Given:
    course_sp     — desired ground-track direction [rad, North=0]
    wind_ne       — wind vector in NE plane [m/s]
    airspeed_sp   — commanded TAS [m/s]

Find heading_sp such that the ground velocity is aligned with course_sp.
Solving the quadratic in ground-speed V_g:
    V_g² - 2·V_g·(W·ĉ) + (|W|² - V_a²) = 0
where ĉ = [cos(course), sin(course)]. Taking the positive root:
    V_g = (W·ĉ) + sqrt(V_a² - (W·ĉ_perp)²)

If crosswind exceeds airspeed (V_a² < (W·ĉ_perp)²), the desired course is
infeasible. We bump airspeed_ref up to make it feasible — sufficient for an
"expert" baseline; in production you'd want PX4's feasibility detection.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


@jax.jit
def course_to_heading_airspeed(
    course_sp: jnp.ndarray,        # scalar [rad]
    wind_ne: jnp.ndarray,          # (2,) wind in NE plane [m/s]
    airspeed_sp: jnp.ndarray,      # scalar [m/s]
    feasibility_margin: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the wind triangle for heading and airspeed reference.

    Args:
        course_sp: Desired ground-track direction [rad, North=0, East=π/2].
        wind_ne: Wind vector in NE plane [N, E] [m/s].
        airspeed_sp: Commanded true airspeed [m/s].
        feasibility_margin: Extra airspeed [m/s] above crosswind floor to keep
            the solution feasible. Default 1.0.

    Returns:
        heading_sp: Aircraft heading required for the desired ground course [rad].
        airspeed_ref: Adjusted airspeed reference (≥ airspeed_sp) [m/s].
    """
    # Course direction (unit vector in NE).
    c_hat = jnp.array([jnp.cos(course_sp), jnp.sin(course_sp)])
    # Perpendicular: rotate +90° (left of course direction).
    c_perp = jnp.array([-jnp.sin(course_sp), jnp.cos(course_sp)])

    # Wind components along / perpendicular to course.
    wind_along = jnp.dot(wind_ne, c_hat)
    wind_cross = jnp.dot(wind_ne, c_perp)

    # If crosswind would make the course infeasible, bump airspeed to keep a
    # finite solution. The +margin term ensures sqrt argument stays positive.
    min_airspeed = jnp.abs(wind_cross) + feasibility_margin
    airspeed_ref = jnp.maximum(airspeed_sp, min_airspeed)

    # Ground speed along course (positive root of the quadratic).
    disc = airspeed_ref ** 2 - wind_cross ** 2
    v_along = jnp.sqrt(jnp.maximum(disc, 0.0))
    v_g = wind_along + v_along

    # Ground velocity in NE — should point in `course_sp` direction by construction.
    vg_ne = v_g * c_hat
    # Air-velocity vector and corresponding heading.
    va_ne = vg_ne - wind_ne
    heading_sp = jnp.arctan2(va_ne[1], va_ne[0])

    return heading_sp, airspeed_ref

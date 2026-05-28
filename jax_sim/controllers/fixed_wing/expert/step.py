"""Expert cascade — top-level composition.

Public API:
    expert_segment_step  — primitive, takes a track segment (PX4 NPFG-shape).
    expert_goto_step     — friendly wrapper, takes a single waypoint.

Both call the same composition path:

    plant_state, wind, segment, alt_sp, V_sp
        │
        ▼
    NPFG (NE plane)          → course_sp
    Wind triangle            → heading_sp, airspeed_ref
    Heading→bank             → roll_sp
    TECS (longitudinal)      → pitch_sp, throttle_sp
    Attitude (P + coord FF)  → rate_sp
    Rate (P+I+FF + anti-WU)  → torques (ail, ele*, rud)
    Trim + slew              → final actuators ∈ [-1,1]⁴ (throttle [0,1])

    * elevator is sign-flipped (mirrors cascade_pid.py:111 — PX4 convention).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jax_sim.controllers.fixed_wing.expert.airspeed_scaling import airspeed_scale  # noqa: F401  (re-export)
from jax_sim.controllers.fixed_wing.expert.attitude import attitude_loop
from jax_sim.controllers.fixed_wing.expert.npfg import npfg_segment_step
from jax_sim.controllers.fixed_wing.expert.rate import rate_loop
from jax_sim.controllers.fixed_wing.expert.slew_rate import slew
from jax_sim.controllers.fixed_wing.expert.tecs import tecs_step
from jax_sim.controllers.fixed_wing.expert.types import (
    ExpertConfig,
    ExpertDebug,
    ExpertState,
)
from jax_sim.controllers.fixed_wing.expert.wind_triangle import course_to_heading_airspeed
from jax_sim.physics.constants import G
from jax_sim.utils.quaternion import quat_to_euler_jax


def _wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap an angle to (-π, π]. Smooth + differentiable everywhere except at π."""
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def _heading_to_roll(
    heading_sp: jnp.ndarray,
    heading: jnp.ndarray,
    ground_speed: jnp.ndarray,
    period: jnp.ndarray,
    max_bank: float = 0.78,   # ~45° — typical FW limit
) -> jnp.ndarray:
    """Bank-angle command from heading error, using coordinated-turn relation.

    desired_yaw_rate ≈ heading_error / period
    lateral_accel    = V_g · desired_yaw_rate
    bank             = atan(lateral_accel / g)

    Clamped to ±max_bank to keep the rate-loop request inside its envelope.
    """
    err = _wrap_to_pi(heading_sp - heading)
    yaw_rate_des = err / jnp.maximum(period, 1e-3)
    lat_accel = ground_speed * yaw_rate_des
    roll_sp = jnp.arctan(lat_accel / G)
    return jnp.clip(roll_sp, -max_bank, max_bank)


@jax.jit
def expert_segment_step(
    plant_state: jnp.ndarray,        # (17,) full plant state
    track_start: jnp.ndarray,        # (3,) NED segment start
    track_end: jnp.ndarray,          # (3,) NED segment end
    airspeed_sp: jnp.ndarray,        # scalar [m/s]
    altitude_sp: jnp.ndarray,        # scalar [m] (positive up)
    expert_state: ExpertState,
    config: ExpertConfig,
    dt: float,
    wind_ned: jnp.ndarray,           # (3,) oracle wind
) -> Tuple[jnp.ndarray, ExpertState, ExpertDebug]:
    """Run one full cascade step.

    Args:
        plant_state: [pos(3), vel(3), quat(4), omega(3), actuators(4)] in NED.
        track_start: Segment start [m, NED]. Only NE used by NPFG; D used for
            altitude_sp by `expert_goto_step` wrapper but ignored here.
        track_end: Segment end [m, NED].
        airspeed_sp: Commanded TAS [m/s].
        altitude_sp: Target altitude above origin [m] (positive up).
        expert_state: Persistent controller state.
        config: Tuned ExpertConfig.
        dt: Timestep [s].
        wind_ned: True wind in NED [m/s].

    Returns:
        actuators: (4,) [ail, ele, rud, thr] — ail/ele/rud ∈ [-1,1], thr ∈ [0,1].
        new_expert_state: Updated controller state.
        debug: Intermediate setpoints (for plotting and future imitation).
    """
    # --- Unpack state -----------------------------------------------------
    pos = plant_state[0:3]
    vel_ned = plant_state[3:6]
    quat = plant_state[6:10]
    omega = plant_state[10:13]

    # Euler angles (aerospace ZYX: roll, pitch, yaw).
    euler = quat_to_euler_jax(quat)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    # Altitude & climb rate in the up-positive convention TECS expects.
    altitude = -pos[2]
    climb_rate = -vel_ned[2]

    # True airspeed = |ground velocity − wind|.
    air_vel_ned = vel_ned - wind_ned
    airspeed = jnp.linalg.norm(air_vel_ned)

    # First-order LPF on airspeed (PX4 does this before feeding gain-scaling).
    alpha_airspeed = dt / jnp.maximum(0.1, dt)  # ≈1 here (tc=0.1); placeholder if we add a tc later
    airspeed_filt = (
        expert_state.airspeed_filt
        + alpha_airspeed * (airspeed - expert_state.airspeed_filt)
    )

    # Landed (NED z >= 0). Multiplicative gate downstream.
    landed = pos[2] >= 0.0

    # --- Lateral guidance: NPFG → course_sp -------------------------------
    course_sp, lat_accel_ff, cross_track = npfg_segment_step(
        pos[0:2],
        vel_ned[0:2],
        track_start[0:2],
        track_end[0:2],
        config,
    )

    # --- Wind triangle: course → heading + airspeed_ref ------------------
    heading_sp, airspeed_ref = course_to_heading_airspeed(
        course_sp, wind_ned[0:2], airspeed_sp
    )

    # --- Heading-to-bank conversion --------------------------------------
    ground_speed = jnp.linalg.norm(vel_ned[0:2])
    roll_sp = _heading_to_roll(
        heading_sp,
        yaw,
        ground_speed,
        config.npfg_period,
    )

    # --- TECS: longitudinal channel --------------------------------------
    # Body-x acceleration FF: we don't have a direct measurement (no IMU sim
    # here), so pass 0. The TECS integrators absorb the bias.
    accel_x_body = jnp.asarray(0.0)
    (pitch_sp, throttle_sp,
     new_int_thr, new_int_pitch,
     new_ste_filt, new_seb_filt) = tecs_step(
        altitude=altitude,
        airspeed=airspeed_filt,
        climb_rate=climb_rate,
        accel_x_body=accel_x_body,
        alt_sp=altitude_sp,
        tas_sp=airspeed_ref,
        tecs_int_throttle=expert_state.tecs_int_throttle,
        tecs_int_pitch=expert_state.tecs_int_pitch,
        tecs_ste_rate_filt=expert_state.tecs_ste_rate_filt,
        tecs_seb_rate_filt=expert_state.tecs_seb_rate_filt,
        config=config,
        dt=dt,
    )

    # --- Attitude: angle → rate setpoint ---------------------------------
    rate_sp, _tilt = attitude_loop(
        roll_sp, pitch_sp, roll, pitch, airspeed_filt, config
    )

    # --- Rate loop: rate → torques ---------------------------------------
    torques, new_rate_int = rate_loop(
        rate_sp,
        omega,
        airspeed_filt,
        expert_state.rate_integral,
        landed,
        config,
        dt,
    )

    # --- Map torques → actuators (with PX4 elevator sign-flip) ----------
    raw_actuators = jnp.array([
        torques[0],              # aileron
        -torques[1],             # elevator (PX4 convention; mirrors cascade_pid.py:111)
        torques[2],              # rudder
        throttle_sp,             # throttle
    ]) + config.trim_actuators

    # Clip to actuator limits per axis.
    raw_actuators = jnp.array([
        jnp.clip(raw_actuators[0], -1.0, 1.0),
        jnp.clip(raw_actuators[1], -1.0, 1.0),
        jnp.clip(raw_actuators[2], -1.0, 1.0),
        jnp.clip(raw_actuators[3], config.throttle_min, config.throttle_max),
    ])

    # --- Slew-rate limit -------------------------------------------------
    actuators = slew(
        expert_state.prev_actuators,
        raw_actuators,
        config.slew_rate,
        dt,
    )

    # --- Pack new state and debug ---------------------------------------
    new_state = ExpertState(
        rate_integral=new_rate_int,
        tecs_int_throttle=new_int_thr,
        tecs_int_pitch=new_int_pitch,
        tecs_ste_rate_filt=new_ste_filt,
        tecs_seb_rate_filt=new_seb_filt,
        airspeed_filt=airspeed_filt,
        prev_actuators=actuators,
        path_anchor=expert_state.path_anchor,
    )

    debug = ExpertDebug(
        course_sp=course_sp,
        heading_sp=heading_sp,
        roll_sp=roll_sp,
        pitch_sp=pitch_sp,
        throttle_sp=throttle_sp,
        rate_sp=rate_sp,
        airspeed_ref=airspeed_ref,
        lateral_accel_ff=lat_accel_ff,
        cross_track_error=cross_track,
    )

    return actuators, new_state, debug


@jax.jit
def expert_goto_step(
    plant_state: jnp.ndarray,
    target_pos: jnp.ndarray,         # (3,) NED
    airspeed_sp: jnp.ndarray,
    expert_state: ExpertState,
    config: ExpertConfig,
    dt: float,
    wind_ned: jnp.ndarray,
) -> Tuple[jnp.ndarray, ExpertState, ExpertDebug]:
    """Single-waypoint wrapper.

    The segment is `(expert_state.path_anchor, target_pos)`. Altitude setpoint
    is derived from the target's NED z-coordinate (`-target_pos[2]`).
    """
    return expert_segment_step(
        plant_state=plant_state,
        track_start=expert_state.path_anchor,
        track_end=target_pos,
        airspeed_sp=airspeed_sp,
        altitude_sp=-target_pos[2],
        expert_state=expert_state,
        config=config,
        dt=dt,
        wind_ned=wind_ned,
    )

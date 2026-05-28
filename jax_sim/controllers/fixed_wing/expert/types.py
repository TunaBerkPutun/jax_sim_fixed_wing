"""ExpertConfig / ExpertState / ExpertDebug NamedTuples + JSON I/O.

All three are immutable JAX pytrees so they can be passed through `jit`,
`vmap`, and `scan` without ceremony. Default values match PX4 fixed-wing
parameter yaml defaults where applicable.

PX4 references (with our defaults):
  rate_kp, rate_ki, rate_ff: FW_RR_*, FW_PR_*, FW_YR_* in
      src/modules/fw_rate_control/fw_rate_control_params.yaml
  rate_imax: FW_RR_IMAX = 0.2, FW_PR_IMAX = 0.4, FW_YR_IMAX = 0.2
  tau_roll, tau_pitch: FW_R_TC, FW_P_TC = 0.4 s (we use 0.4 by default)
  airspeed_trim: FW_AIRSPD_TRIM = 15 m/s
  TECS: FW_T_* family
  NPFG: NPFG_PERIOD = 1.0 s, NPFG_DAMPING = 0.7
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class ExpertConfig(NamedTuple):
    """Fixed gains, limits, and structural parameters for the expert cascade."""

    # --- Rate loop (per-axis [roll, pitch, yaw]) -----------------------------
    rate_kp: jnp.ndarray  # (3,) PX4 FW_RR_P / FW_PR_P / FW_YR_P
    rate_ki: jnp.ndarray  # (3,) FW_RR_I / FW_PR_I / FW_YR_I
    rate_ff: jnp.ndarray  # (3,) FW_RR_FF / FW_PR_FF / FW_YR_FF
    rate_imax: jnp.ndarray  # (3,) FW_RR_IMAX / FW_PR_IMAX / FW_YR_IMAX
    rate_limit: jnp.ndarray  # (3,) FW_R_RMAX / FW_P_RMAX_POS / FW_Y_RMAX  [rad/s]
    airspeed_trim: float  # FW_AIRSPD_TRIM [m/s]
    airspeed_scale_min: float  # FW_ARSP_SCALE_EN clamp lo
    airspeed_scale_max: float  # FW_ARSP_SCALE_EN clamp hi

    # --- Attitude loop -------------------------------------------------------
    tau_roll: float  # FW_R_TC [s]
    tau_pitch: float  # FW_P_TC [s]
    coord_turn_gate_lo: float  # full FF below this |angle| [rad]
    coord_turn_gate_hi: float  # zero FF above this |angle| [rad]

    # --- TECS-lite -----------------------------------------------------------
    tecs_alt_p: float  # FW_T_ALT_TC inverse — altitude error → climb_rate_sp
    tecs_thr_p: float  # specific energy rate → throttle proportional
    tecs_thr_i: float  # specific energy rate → throttle integral
    tecs_pitch_p: float  # specific energy balance rate → pitch proportional
    tecs_pitch_i: float  # specific energy balance rate → pitch integral
    tecs_speed_weight: float  # 0..2; 1.0 balanced (FW_T_SPDWEIGHT)
    tecs_climb_rate_max: float  # FW_T_CLMB_R_SP [m/s]
    tecs_sink_rate_max: float  # FW_T_SINK_R_SP [m/s]
    tecs_pitch_min: float  # FW_P_LIM_MIN [rad]
    tecs_pitch_max: float  # FW_P_LIM_MAX [rad]
    tecs_vert_accel_max: float  # FW_T_VERT_ACC [m/s²]
    tecs_ste_tc: float  # low-pass tc for STE rate [s]
    tecs_seb_tc: float  # low-pass tc for SEB rate [s]
    tecs_throttle_trim: float  # FW_THR_TRIM [0..1]

    # --- NPFG-lite -----------------------------------------------------------
    npfg_period: float  # NPFG_PERIOD [s]
    npfg_damping: float  # NPFG_DAMPING (zeta)

    # --- Output stage --------------------------------------------------------
    trim_actuators: jnp.ndarray  # (4,) [ail_trim, ele_trim, rud_trim, thr_trim]
    slew_rate: jnp.ndarray  # (4,) per-actuator max slew [units/s]
    throttle_min: float
    throttle_max: float


# -----------------------------------------------------------------------------
# Controller state
# -----------------------------------------------------------------------------


class ExpertState(NamedTuple):
    """Persistent controller state — explicit pytree, no hidden globals."""

    # Rate loop integrators (one per axis)
    rate_integral: jnp.ndarray  # (3,)

    # TECS state
    tecs_int_throttle: float  # STE-rate integrator
    tecs_int_pitch: float  # SEB-rate integrator
    tecs_ste_rate_filt: float  # low-passed STE rate
    tecs_seb_rate_filt: float  # low-passed SEB rate

    # Airspeed filter (PX4 LPFs airspeed before consuming it)
    airspeed_filt: float

    # Slew-rate limiter memory
    prev_actuators: jnp.ndarray  # (4,)

    # Used by `expert_goto_step` wrapper to synthesize a segment
    path_anchor: jnp.ndarray  # (3,) NED


# -----------------------------------------------------------------------------
# Debug output (intermediate setpoints — useful for plotting & imitation)
# -----------------------------------------------------------------------------


class ExpertDebug(NamedTuple):
    course_sp: float
    heading_sp: float
    roll_sp: float
    pitch_sp: float
    throttle_sp: float
    rate_sp: jnp.ndarray  # (3,)
    airspeed_ref: float
    lateral_accel_ff: float
    cross_track_error: float


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


def default_expert_config() -> ExpertConfig:
    """PX4 fixed-wing defaults where applicable.

    Rate gains here are placeholders (PX4 yaml defaults). Run
    `scripts/tune_expert.py` to produce a config tuned for *this* airframe via
    jacrev linearization; the result is written to `tuned_expert_config.json`.
    """
    return ExpertConfig(
        # Rate loop — PX4 defaults from fw_rate_control_params.yaml
        rate_kp=jnp.array([0.05, 0.08, 0.05]),
        rate_ki=jnp.array([0.1, 0.1, 0.1]),
        rate_ff=jnp.array([0.5, 0.5, 0.3]),
        rate_imax=jnp.array([0.2, 0.4, 0.2]),
        rate_limit=jnp.array([jnp.deg2rad(70.0), jnp.deg2rad(60.0), jnp.deg2rad(50.0)]),
        airspeed_trim=15.0,
        airspeed_scale_min=0.5,
        airspeed_scale_max=2.0,
        # Attitude loop — PX4 defaults
        tau_roll=0.4,
        tau_pitch=0.4,
        coord_turn_gate_lo=jnp.deg2rad(70.0),
        coord_turn_gate_hi=jnp.deg2rad(75.0),
        # TECS-lite — reasonable defaults
        tecs_alt_p=0.5,
        tecs_thr_p=0.1,
        tecs_thr_i=0.1,
        tecs_pitch_p=0.1,
        tecs_pitch_i=0.05,
        tecs_speed_weight=1.0,
        tecs_climb_rate_max=5.0,
        tecs_sink_rate_max=2.5,
        tecs_pitch_min=jnp.deg2rad(-30.0),
        tecs_pitch_max=jnp.deg2rad(25.0),
        tecs_vert_accel_max=7.0,
        tecs_ste_tc=0.5,
        tecs_seb_tc=0.5,
        tecs_throttle_trim=0.6,
        # NPFG — PX4 defaults
        npfg_period=1.0,
        npfg_damping=0.7,
        # Output stage
        trim_actuators=jnp.zeros(4),
        slew_rate=jnp.array([1e6, 1e6, 1e6, 1e6]),  # effectively off
        throttle_min=0.0,
        throttle_max=1.0,
    )


def init_expert_state(
    initial_plant_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    initial_airspeed: float = 20.0,
) -> ExpertState:
    """Construct a fresh ExpertState.

    The `path_anchor` is set to the current XY plane position so the goto
    wrapper has a sensible segment start. The altitude axis of the anchor is
    set to the current altitude (mirroring PX4 behaviour where the active
    segment starts at the aircraft's current position when a new leg begins).
    """
    pos = initial_plant_state[0:3]
    return ExpertState(
        rate_integral=jnp.zeros(3),
        tecs_int_throttle=0.0,
        tecs_int_pitch=0.0,
        tecs_ste_rate_filt=0.0,
        tecs_seb_rate_filt=0.0,
        airspeed_filt=initial_airspeed,
        prev_actuators=initial_plant_state[13:17],
        path_anchor=pos,
    )


# -----------------------------------------------------------------------------
# JSON I/O
# -----------------------------------------------------------------------------


def expert_config_to_dict(config: ExpertConfig) -> dict:
    """Convert ExpertConfig to a JSON-serializable dict."""
    out: dict = {}
    for field in config._fields:
        value = getattr(config, field)
        if hasattr(value, "ndim") and value.ndim >= 1:
            out[field] = [float(x) for x in value]
        else:
            out[field] = float(value)
    return out


def expert_config_from_dict(data: dict) -> ExpertConfig:
    """Construct ExpertConfig from a dict (e.g. loaded JSON)."""
    array_fields = {"rate_kp", "rate_ki", "rate_ff", "rate_imax", "rate_limit",
                    "trim_actuators", "slew_rate"}
    kwargs = {}
    for field in ExpertConfig._fields:
        value = data[field]
        if field in array_fields:
            kwargs[field] = jnp.array(value)
        else:
            kwargs[field] = float(value)
    return ExpertConfig(**kwargs)


def save_expert_config(config: ExpertConfig, filepath: str | Path) -> None:
    """Write ExpertConfig to JSON."""
    with open(filepath, "w") as f:
        json.dump(expert_config_to_dict(config), f, indent=2)


def load_expert_config(filepath: str | Path) -> ExpertConfig:
    """Load ExpertConfig from JSON."""
    with open(filepath, "r") as f:
        return expert_config_from_dict(json.load(f))

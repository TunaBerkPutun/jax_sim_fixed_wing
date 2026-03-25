"""Wind and turbulence models for fixed-wing simulation."""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from jax_sim.utils.quaternion import quat_inv, rotate_vec_by_quat


@flax.struct.dataclass
class WindConfig:
    """Configuration for steady wind, gusts, and Dryden-style turbulence."""

    steady_wind_ned: jnp.ndarray
    enable_gust: float
    gust_direction_ned: jnp.ndarray
    gust_magnitude: float
    gust_start_time: float
    gust_rise_time: float
    gust_hold_time: float
    enable_turbulence: float
    turbulence_sigma: jnp.ndarray
    turbulence_length_scale: jnp.ndarray


def create_wind_config(
    steady_wind_ned: jnp.ndarray | None = None,
    enable_gust: bool = False,
    gust_direction_ned: jnp.ndarray | None = None,
    gust_magnitude: float = 0.0,
    gust_start_time: float = 5.0,
    gust_rise_time: float = 2.0,
    gust_hold_time: float = 2.0,
    enable_turbulence: bool = False,
    turbulence_sigma: jnp.ndarray | None = None,
    turbulence_length_scale: jnp.ndarray | None = None,
) -> WindConfig:
    """Create wind configuration with safe defaults."""
    if steady_wind_ned is None:
        steady_wind_ned = jnp.zeros(3)
    if gust_direction_ned is None:
        gust_direction_ned = jnp.array([1.0, 0.0, 0.0])
    if turbulence_sigma is None:
        turbulence_sigma = jnp.array([1.0, 1.0, 0.5])
    if turbulence_length_scale is None:
        turbulence_length_scale = jnp.array([200.0, 200.0, 50.0])

    return WindConfig(
        steady_wind_ned=jnp.asarray(steady_wind_ned, dtype=jnp.float32),
        enable_gust=jnp.asarray(1.0 if enable_gust else 0.0, dtype=jnp.float32),
        gust_direction_ned=jnp.asarray(gust_direction_ned, dtype=jnp.float32),
        gust_magnitude=jnp.asarray(gust_magnitude, dtype=jnp.float32),
        gust_start_time=jnp.asarray(gust_start_time, dtype=jnp.float32),
        gust_rise_time=jnp.asarray(gust_rise_time, dtype=jnp.float32),
        gust_hold_time=jnp.asarray(gust_hold_time, dtype=jnp.float32),
        enable_turbulence=jnp.asarray(1.0 if enable_turbulence else 0.0, dtype=jnp.float32),
        turbulence_sigma=jnp.asarray(turbulence_sigma, dtype=jnp.float32),
        turbulence_length_scale=jnp.asarray(turbulence_length_scale, dtype=jnp.float32),
    )


DEFAULT_WIND_CONFIG = create_wind_config()


@jax.jit
def _safe_unit(vec: jnp.ndarray) -> jnp.ndarray:
    """Normalize vector with zero guard."""
    norm = jnp.linalg.norm(vec)
    safe_norm = jnp.where(norm > 1e-6, norm, 1.0)
    unit = vec / safe_norm
    return jnp.where(norm > 1e-6, unit, jnp.array([1.0, 0.0, 0.0]))


@jax.jit
def one_minus_cosine_gust_scale(
    time: float,
    start_time: float,
    rise_time: float,
    hold_time: float,
) -> float:
    """Compute one-minus-cosine gust scale in [0, 1]."""
    rise = jnp.maximum(rise_time, 1e-3)
    hold = jnp.maximum(hold_time, 0.0)

    t1 = start_time
    t2 = t1 + rise
    t3 = t2 + hold
    t4 = t3 + rise

    tau_rise = jnp.clip((time - t1) / rise, 0.0, 1.0)
    tau_fall = jnp.clip((time - t3) / rise, 0.0, 1.0)
    rise_profile = 0.5 * (1.0 - jnp.cos(jnp.pi * tau_rise))
    fall_profile = 0.5 * (1.0 + jnp.cos(jnp.pi * tau_fall))

    return jnp.where(
        time < t1,
        0.0,
        jnp.where(
            time < t2,
            rise_profile,
            jnp.where(time < t3, 1.0, jnp.where(time < t4, fall_profile, 0.0)),
        ),
    )


@jax.jit
def compute_gust_ned(time: float, wind_config: WindConfig) -> jnp.ndarray:
    """Compute one-minus-cosine gust vector in NED frame."""
    direction = _safe_unit(wind_config.gust_direction_ned)
    scale = one_minus_cosine_gust_scale(
        time=time,
        start_time=wind_config.gust_start_time,
        rise_time=wind_config.gust_rise_time,
        hold_time=wind_config.gust_hold_time,
    )
    return wind_config.enable_gust * direction * wind_config.gust_magnitude * scale


@jax.jit
def update_dryden_turbulence(
    turbulence_ned: jnp.ndarray,
    airspeed: float,
    key,
    dt: float,
    wind_config: WindConfig,
) -> jnp.ndarray:
    """Update Dryden-style turbulence state using an OU process per axis."""
    speed = jnp.maximum(airspeed, 1.0)
    length_scale = jnp.maximum(wind_config.turbulence_length_scale, 1.0)
    sigma = jnp.maximum(wind_config.turbulence_sigma, 0.0)

    a = speed / length_scale
    phi = jnp.exp(-a * dt)
    xi = jax.random.normal(key, shape=(3,))
    q = sigma * jnp.sqrt(jnp.maximum(1.0 - phi * phi, 0.0))

    next_turbulence = phi * turbulence_ned + q * xi
    return wind_config.enable_turbulence * next_turbulence


@jax.jit
def compute_total_wind_ned(
    time: float,
    turbulence_ned: jnp.ndarray,
    wind_config: WindConfig,
) -> jnp.ndarray:
    """Compute total wind in NED frame."""
    gust_ned = compute_gust_ned(time, wind_config)
    turbulence_term = wind_config.enable_turbulence * turbulence_ned
    return wind_config.steady_wind_ned + gust_ned + turbulence_term


@jax.jit
def step_wind_model(
    plane_state: jnp.ndarray,
    turbulence_ned: jnp.ndarray,
    time: float,
    key,
    dt: float,
    wind_config: WindConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance turbulence state and return total wind in NED frame."""
    speed = jnp.linalg.norm(plane_state[3:6])
    next_turbulence_ned = update_dryden_turbulence(
        turbulence_ned=turbulence_ned,
        airspeed=speed,
        key=key,
        dt=dt,
        wind_config=wind_config,
    )
    total_wind_ned = compute_total_wind_ned(time, next_turbulence_ned, wind_config)
    return next_turbulence_ned, total_wind_ned


@jax.jit
def wind_ned_to_body(quat: jnp.ndarray, wind_ned: jnp.ndarray) -> jnp.ndarray:
    """Rotate NED wind vector into body frame."""
    return rotate_vec_by_quat(quat_inv(quat), wind_ned)

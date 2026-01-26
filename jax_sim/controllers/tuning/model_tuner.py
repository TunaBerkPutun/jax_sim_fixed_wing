"""Model-based tuning using linearization of the simulator."""

from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
from jax import random

from jax_sim.controllers.pid_gains import PIDConfig, create_pid_config
from jax_sim.controllers.tuning.es_tuner import TuningResult
from jax_sim.controllers.tuning.loss import (
    RATE_PARAM_BOUNDS,
    config_to_params,
    config_to_rate_params,
    evaluate_rate_config,
)
from jax_sim.physics.constants import FLAP_MAX, Inertia, Inertia_inv
from jax_sim.physics.dynamics import get_forces_and_moments


def _trim_state(throttle_cmd: float) -> jnp.ndarray:
    """Return trim state for linearization."""
    return jnp.array([
        0.0, 0.0, -100.0,       # Position (NED)
        20.0, 0.0, 0.0,         # Velocity
        1.0, 0.0, 0.0, 0.0,     # Quaternion (level)
        0.0, 0.0, 0.0,          # Angular velocity
        0.0, 0.0, 0.0, throttle_cmd,  # Actuator states
    ])


def _omega_dot(state: jnp.ndarray, actuators: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time angular acceleration from state and actuators."""
    _, M_body, _ = get_forces_and_moments(state, actuators)
    omega = state[10:13]
    term_gyroscopic = jnp.cross(omega, Inertia @ omega)
    return Inertia_inv @ (M_body - term_gyroscopic)


def _rate_dynamics_from_linearization(
    throttle_cmds: Iterable[float],
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (a, b) for omega dynamics across throttle regimes."""
    A_list = []
    B_list = []
    for throttle_cmd in throttle_cmds:
        trim_state = _trim_state(float(throttle_cmd))
        base_actuators = jnp.array([0.0, 0.0, 0.0, float(throttle_cmd)])

        def omega_dot_from_omega(omega):
            state = trim_state.at[10:13].set(omega)
            return _omega_dot(state, base_actuators)

        def omega_dot_from_cmd(cmd):
            actuators = jnp.array([
                cmd[0] * FLAP_MAX,
                -cmd[1] * FLAP_MAX,
                cmd[2] * FLAP_MAX,
                float(throttle_cmd),
            ])
            return _omega_dot(trim_state, actuators)

        A_omega = jax.jacrev(omega_dot_from_omega)(jnp.zeros(3))
        B_omega = jax.jacrev(omega_dot_from_cmd)(jnp.zeros(3))

        A_list.append(A_omega)
        B_list.append(B_omega)

    A_omega = jnp.stack(A_list)
    B_omega = jnp.stack(B_list)

    a = jnp.diagonal(A_omega, axis1=-2, axis2=-1)
    b = jnp.diagonal(B_omega, axis1=-2, axis2=-1)
    return a, b


def _select_rate_wn(
    a: jnp.ndarray,
    b: jnp.ndarray,
    wn: float,
    zeta: float,
    kp_target: float,
    kp_margin: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Select per-axis wn to meet stability and a minimum Kp target."""
    b_sign = jnp.where(b >= 0.0, 1.0, -1.0)
    b_safe = jnp.where(jnp.abs(b) < 1e-6, b_sign * 1e-6, b)
    kp_target = jnp.maximum(kp_target, 0.0)
    kp_target_signed = kp_target * jnp.sign(b_safe)
    kp_req = kp_target_signed

    stable_kp = (-a / b_safe) + jnp.sign(b_safe) * kp_margin
    needs_stability = (a + b_safe * kp_req) <= 0.0
    kp_req = jnp.where(needs_stability, stable_kp, kp_req)

    wn_req = (b_safe * kp_req - a) / (2.0 * zeta)
    wn_used = jnp.maximum(wn_req, wn)
    return wn_used, kp_req


def _design_pi(
    a: jnp.ndarray,
    b: jnp.ndarray,
    wn: jnp.ndarray,
    zeta: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Design PI gains for first-order plant: p_dot = a*p + b*u."""
    b_sign = jnp.where(b >= 0.0, 1.0, -1.0)
    b_safe = jnp.where(jnp.abs(b) < 1e-6, b_sign * 1e-6, b)
    kp = (2.0 * zeta * wn + a) / b_safe
    ki = (wn ** 2) / b_safe
    return kp, ki


def run_model_tuning_rate(
    throttle_cmds: Iterable[float] = (0.2, 0.9),
    dt: float = 0.004,
    wn: float = 8.0,
    zeta: float = 0.7,
    kp_target: float = 0.2,
    kp_margin: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> TuningResult:
    """Tune rate gains by linearizing the simulator around trim."""
    throttle_cmds = tuple(throttle_cmds)
    a, b = _rate_dynamics_from_linearization(throttle_cmds, dt)

    a_mean = jnp.mean(a, axis=0)
    b_mean = jnp.mean(b, axis=0)

    wn_used, kp_req = _select_rate_wn(a_mean, b_mean, wn, zeta, kp_target, kp_margin)
    rate_kp, rate_ki = _design_pi(a_mean, b_mean, wn_used, zeta)
    rate_kd = jnp.zeros(3)

    rate_kp = jnp.clip(rate_kp, RATE_PARAM_BOUNDS[0:3, 0], RATE_PARAM_BOUNDS[0:3, 1])
    rate_ki = jnp.clip(rate_ki, RATE_PARAM_BOUNDS[3:6, 0], RATE_PARAM_BOUNDS[3:6, 1])
    rate_kd = jnp.clip(rate_kd, RATE_PARAM_BOUNDS[6:9, 0], RATE_PARAM_BOUNDS[6:9, 1])

    base = create_pid_config()
    config = PIDConfig(
        tau_roll=base.tau_roll,
        tau_pitch=base.tau_pitch,
        rate_kp=rate_kp,
        rate_ki=rate_ki,
        rate_kd=rate_kd,
        speed_kp=base.speed_kp,
        speed_ki=base.speed_ki,
        throttle_ff=base.throttle_ff,
        rate_limit=base.rate_limit,
        integral_limit=base.integral_limit,
    )

    key = random.PRNGKey(seed)
    final_loss = float(evaluate_rate_config(config_to_rate_params(config), key))

    if verbose:
        print("Model-based rate tuning:")
        print(f"  Throttle regimes: {throttle_cmds}")
        print(f"  a (mean): {a_mean}")
        print(f"  b (mean): {b_mean}")
        print(f"  wn(target)={wn:.2f}, wn(used)={wn_used}")
        print(f"  kp_target={kp_target:.3f}, kp_min_used={kp_req}")
        print(f"  zeta={zeta:.2f}")

    return TuningResult(
        params=config_to_params(config),
        config=config,
        final_loss=final_loss,
        loss_history=jnp.array([final_loss]),
    )

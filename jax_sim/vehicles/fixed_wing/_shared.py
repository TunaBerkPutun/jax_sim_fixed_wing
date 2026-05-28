"""Shared helpers across fixed-wing tiers (spec §6, §9.3).

Pure JAX. No I/O. No tier1/tier0 imports at module-load time except
`tier1.forces_moments`, which is the canonical aero+thrust side-export
that Tier 0 linearizes around (spec §9.3).
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing import tier1 as _tier1
from jax_sim.vehicles.fixed_wing.params import FixedWingParams
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_FIXED_WING


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------

def _trim_residual(
    unknowns: jnp.ndarray,
    params: FixedWingParams,
    airspeed: float,
    altitude: float,
) -> jnp.ndarray:
    """3-residual system for steady wings-level flight.

    Unknowns: (alpha, ele_norm, throttle).
    Residuals: (Fx_body + Wx_body, Fz_body + Wz_body, My_body) with the
    quaternion set so that pitch = alpha (body velocity = [V cos a, 0, V sin a]
    in body frame; earth-frame velocity is [V, 0, 0] — i.e. no descent).
    """
    alpha = unknowns[0]
    ele_norm = unknowns[1]
    thr = unknowns[2]

    flap_max = params.actuators.flap_max
    mass = params.mass_props.mass
    g = params.environment.gravity

    # Body-frame velocity at alpha (level flight, no sideslip)
    u = airspeed * jnp.cos(alpha)
    w = airspeed * jnp.sin(alpha)
    v_earth = jnp.array([airspeed, 0.0, 0.0])  # level NED velocity

    # Pitch-only quaternion (rotate by alpha about body-y)
    half = 0.5 * alpha
    quat = jnp.array([jnp.cos(half), 0.0, jnp.sin(half), 0.0])

    # Actuators in canonical units (radians for surfaces, [0,1] throttle)
    actuators = jnp.array([0.0, ele_norm * flap_max, 0.0, thr])

    # State vector matching tier1's (17,) convention
    state = jnp.concatenate([
        jnp.array([0.0, 0.0, -altitude]),  # pos
        v_earth,
        quat,
        jnp.zeros(3),                       # omega
        actuators,
    ])

    F, M = _tier1.forces_moments(state, params, jnp.zeros(3))

    # Gravity in body frame: rotate earth gravity [0,0,mg] back via inverse quat.
    # For wings-level pitch-only attitude:
    Wx = -mass * g * jnp.sin(alpha)
    Wz = mass * g * jnp.cos(alpha)

    res_fx = F[0] + Wx
    res_fz = F[2] + Wz
    res_my = M[1]
    return jnp.array([res_fx, res_fz, res_my])


def solve_trim(
    params: FixedWingParams = DEFAULT_FIXED_WING,
    airspeed: float = 20.0,
    altitude: float = 100.0,
    n_newton_iters: int = 20,
):
    """Solve for steady wings-level flight via pure-JAX Newton.

    Returns:
        trim_state:     (17,) — pos at altitude, vel = [V, 0, 0] NED,
                        pitch-only quat (theta = alpha), zero omega,
                        actuators = [0, ele_rad, 0, throttle].
        trim_actuators: (4,) — user_cmd in tier1 input convention
                        ([-1,1] for surfaces, [0,1] for throttle). At steady
                        state actuators have settled to the user_cmd target, so
                        user_cmd_trim = [0, ele_norm, 0, throttle].
    """
    x = jnp.array([jnp.deg2rad(2.0), 0.0, 0.3])

    def body_fn(_, x):
        res = _trim_residual(x, params, airspeed, altitude)
        J = jax.jacrev(lambda y: _trim_residual(y, params, airspeed, altitude))(x)
        # Solve J·dx = -res; jnp.linalg.solve is fine for 3x3.
        dx = jnp.linalg.solve(J, -res)
        return x + dx

    x = jax.lax.fori_loop(0, n_newton_iters, body_fn, x)

    alpha = x[0]
    ele_norm = x[1]
    thr = x[2]
    flap_max = params.actuators.flap_max

    half = 0.5 * alpha
    quat = jnp.array([jnp.cos(half), 0.0, jnp.sin(half), 0.0])
    actuators_state = jnp.array([0.0, ele_norm * flap_max, 0.0, thr])

    trim_state = jnp.concatenate([
        jnp.array([0.0, 0.0, -altitude]),
        jnp.array([airspeed, 0.0, 0.0]),
        quat,
        jnp.zeros(3),
        actuators_state,
    ])

    # user_cmd convention: surfaces in [-1,1], throttle in [0,1].
    # At trim, settled actuator state == target (no residual lag), so
    # the user_cmd that maintains trim equals the actuator state in normalized
    # form: [ail_norm, ele_norm, rud_norm, throttle].
    trim_actuators = jnp.array([0.0, ele_norm, 0.0, thr])

    return trim_state, trim_actuators


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class Tier0Coeffs:
    """Linear-coefficient buildup around a trim point (spec §4.1, §9.3).

    Linearizes the side-export `tier1.forces_moments` plus thrust:
        [F; M](state, act) ≈ [F_trim; M_trim]
                              + A @ (state - trim_state)
                              + B @ (act   - trim_actuators_radians)

    Note `B`'s columns correspond to the *actuator-state* slice of the (17,)
    state vector (radians for surfaces, [0,1] throttle), since that is the
    representation tier1.forces_moments differentiates against. The Tier 0
    `equations_of_motion` reuses tier1.update_actuators (which converts the
    user_cmd in [-1,1] to radians via flap_max), so the user-visible cmd shape
    is unchanged.
    """
    A: jnp.ndarray         # (6, 17)
    B: jnp.ndarray         # (6, 4)
    F_trim: jnp.ndarray    # (3,)
    M_trim: jnp.ndarray    # (3,)
    trim_state: jnp.ndarray         # (17,)
    trim_actuators: jnp.ndarray     # (4,) — user_cmd convention ([-1,1] / [0,1])


def _stacked_fm(state: jnp.ndarray, actuators: jnp.ndarray,
                params: FixedWingParams, wind_body: jnp.ndarray) -> jnp.ndarray:
    """[F; M] as a single (6,) vector for one-shot jacrev."""
    # Inject `actuators` into the actuator slice so jacrev w.r.t. `actuators`
    # gives a clean (6, 4) Jacobian.
    state_with_act = jnp.concatenate([state[:13], actuators])
    F, M = _tier1.forces_moments(state_with_act, params, wind_body)
    return jnp.concatenate([F, M])


def extract_tier0_coeffs(
    params: FixedWingParams,
    trim_state: jnp.ndarray,
    trim_actuators_usercmd: jnp.ndarray,
    wind_body: jnp.ndarray = jnp.zeros(3),
) -> Tier0Coeffs:
    """Linearize `tier1.forces_moments` around the trim point.

    `trim_actuators_usercmd` is the user_cmd convention ([-1,1] for surfaces,
    [0,1] for throttle); internally it is converted to the radians/throttle
    actuator-state convention used by `tier1.forces_moments`.
    """
    flap_max = params.actuators.flap_max
    # Convert user_cmd → actuator-state radians for the Jacobian centerpoint.
    act_state_trim = jnp.array([
        trim_actuators_usercmd[0] * flap_max,
        trim_actuators_usercmd[1] * flap_max,
        trim_actuators_usercmd[2] * flap_max,
        trim_actuators_usercmd[3],
    ])

    # A: ∂[F;M]/∂state at trim. The state argument is the full (17,) vector;
    # the actuator slice is held by `actuators_state` for B's Jacobian, so we
    # build a wrapper that overrides only the position/vel/quat/omega part.
    def fm_state(state17):
        return _stacked_fm(state17, act_state_trim, params, wind_body)

    def fm_act(act4):
        return _stacked_fm(trim_state, act4, params, wind_body)

    A = jax.jacrev(fm_state)(trim_state)
    B = jax.jacrev(fm_act)(act_state_trim)

    F_trim, M_trim = _tier1.forces_moments(
        jnp.concatenate([trim_state[:13], act_state_trim]),
        params,
        wind_body,
    )

    return Tier0Coeffs(
        A=A,
        B=B,
        F_trim=F_trim,
        M_trim=M_trim,
        trim_state=trim_state,
        trim_actuators=trim_actuators_usercmd,
    )


__all__ = [
    "Tier0Coeffs",
    "solve_trim",
    "extract_tier0_coeffs",
]

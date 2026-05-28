"""Fixed-wing Tier 0 — linear coefficient buildup (spec §4.1, §9.3).

Tier 0 is the `jax.jacrev` linearization of Tier 1 around a trim point. Same
Vehicle Module Contract (`init_state`, `forces_moments`, `step`) as Tier 1 so
controllers can swap tiers without code changes.

Wind is **ignored** by Tier 0 — the linearization is anchored at trim wind
(default zero). For non-trivial wind behavior, use Tier 1 or re-linearize at
the operating wind.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from jax_sim.vehicles.fixed_wing.params import FixedWingParams
from jax_sim.vehicles.fixed_wing.presets import DEFAULT_FIXED_WING
from jax_sim.vehicles.fixed_wing import tier1 as _tier1
from jax_sim.vehicles.fixed_wing._shared import (
    Tier0Coeffs,
    solve_trim,
    extract_tier0_coeffs,
)
from jax_sim.physics.rigid_body import rigid_body_step


@flax.struct.dataclass
class FixedWingTier0Params:
    """Tier 0 parameter bundle.

    `coeffs` holds the linearization (anchored at `trim_state`,
    `trim_actuators`); `base_params` carries actuator lag, mass, environment,
    and propulsion limits that the integrator still needs from Tier 1.
    """
    coeffs: Tier0Coeffs
    base_params: FixedWingParams


def create_default_tier0(
    airspeed: float = 20.0,
    altitude: float = 100.0,
    base_params: FixedWingParams = DEFAULT_FIXED_WING,
) -> FixedWingTier0Params:
    """Build the default Tier 0 params (PX4 SIH UAV, V=20 m/s cruise)."""
    trim_s, trim_a = solve_trim(base_params, airspeed=airspeed, altitude=altitude)
    coeffs = extract_tier0_coeffs(base_params, trim_s, trim_a)
    return FixedWingTier0Params(coeffs=coeffs, base_params=base_params)


DEFAULT_FIXED_WING_TIER0 = create_default_tier0()


# ---------------------------------------------------------------------------
# Vehicle Module Contract entrypoints (spec §7.3)
# ---------------------------------------------------------------------------

def init_state(
    pos: jnp.ndarray = jnp.zeros(3),
    vel: jnp.ndarray = jnp.array([20.0, 0.0, 0.0]),
    quat: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0]),
    omega: jnp.ndarray = jnp.zeros(3),
    actuators: jnp.ndarray = jnp.zeros(4),
) -> jnp.ndarray:
    """Build a valid (17,) fixed-wing state. Identical to tier1.init_state."""
    return jnp.concatenate([pos, vel, quat, omega, actuators])


@jax.jit
def forces_moments(
    state: jnp.ndarray,
    params: FixedWingTier0Params = DEFAULT_FIXED_WING_TIER0,
    wind_body: jnp.ndarray = jnp.zeros(3),
):
    """Linear forces/moments around trim.

    F = F_trim + A_F · (state - trim_state) + B_F · (act_state - act_state_trim)
    M = M_trim + A_M · (state - trim_state) + B_M · (act_state - act_state_trim)

    `act_state` is the actuator slice of `state` (radians / [0,1]); the
    linearization B was built in that frame so we use it directly.
    """
    del wind_body  # Tier 0 ignores wind (anchored at trim wind).
    c = params.coeffs
    flap_max = params.base_params.actuators.flap_max

    delta_state = state - c.trim_state
    act_state = state[13:17]
    act_state_trim = jnp.array([
        c.trim_actuators[0] * flap_max,
        c.trim_actuators[1] * flap_max,
        c.trim_actuators[2] * flap_max,
        c.trim_actuators[3],
    ])
    delta_act = act_state - act_state_trim

    FM_trim = jnp.concatenate([c.F_trim, c.M_trim])
    FM = FM_trim + c.A @ delta_state + c.B @ delta_act
    return FM[:3], FM[3:]


@jax.jit
def equations_of_motion(
    state: jnp.ndarray,
    user_commands: jnp.ndarray,
    dt: float = 0.004,
    params: FixedWingTier0Params = DEFAULT_FIXED_WING_TIER0,
    wind_body: jnp.ndarray = jnp.zeros(3),
) -> jnp.ndarray:
    """One Tier 0 step. Same outer shape as `tier1.equations_of_motion`.

    1. First-order actuator lag (reused from tier1).
    2. Linearized F, M from `forces_moments` evaluated *after* the actuator
       update — matches tier1's evaluation order.
    3. RK4 rigid-body integration (vehicle-agnostic).
    4. Body-rate clip and ground-contact reset, identical to tier1.
    """
    del wind_body  # Tier 0 anchored at trim wind.
    base = params.base_params

    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    current_actuators = state[13:17]

    new_actuators = _tier1.update_actuators(
        current_actuators=current_actuators,
        user_commands=user_commands,
        dt=dt,
        aircraft=base,
    )

    # Build the state passed to forces_moments (with refreshed actuators).
    state_for_fm = jnp.concatenate([pos, vel, quat, omega, new_actuators])
    F_body, M_body = forces_moments(state_for_fm, params, jnp.zeros(3))

    new_pos, new_vel, new_quat, new_omega = rigid_body_step(
        pos=pos,
        vel=vel,
        quat=quat,
        omega=omega,
        F_body=F_body,
        M_body=M_body,
        dt=dt,
        mass_props=base.mass_props,
        environment=base.environment,
        max_body_rate=base.actuators.max_body_rate,
    )

    next_state = jnp.concatenate([new_pos, new_vel, new_quat, new_omega, new_actuators])

    landed = new_pos[2] >= 0.0
    next_state = jax.lax.select(
        landed,
        jnp.concatenate([new_pos, jnp.zeros(3), new_quat, jnp.zeros(3), jnp.zeros(4)]),
        next_state,
    )
    return next_state


# `step` matches the spec's uniform signature: (state, cmd, dt, params, wind_body)
step = equations_of_motion


__all__ = [
    "FixedWingTier0Params",
    "Tier0Coeffs",
    "create_default_tier0",
    "DEFAULT_FIXED_WING_TIER0",
    "init_state",
    "forces_moments",
    "step",
    "equations_of_motion",
]

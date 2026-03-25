#!/usr/bin/env python3
"""Test script for wind, gust, and turbulence models."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_sim.env.wrappers import make_env
from jax_sim.physics.dynamics import compute_aircraft_forces_moments
from jax_sim.physics.wind import create_wind_config, compute_total_wind_ned, step_wind_model


def _initial_state() -> jnp.ndarray:
    """Create nominal level-flight state."""
    return jnp.array(
        [
            0.0, 0.0, -100.0,  # Position (NED)
            20.0, 0.0, 0.0,    # Velocity (NED)
            1.0, 0.0, 0.0, 0.0,  # Quaternion
            0.0, 0.0, 0.0,     # Angular velocity
            0.0, 0.0, 0.0, 0.5,  # Actuator states
        ]
    )


def test_steady_wind_aero_effect():
    """Headwind should increase airspeed and drag compared to calm/tailwind."""
    print("\n=== TEST: Steady Wind Aero Effect ===")
    state = _initial_state()
    actuators = state[13:17]

    calm_wind = jnp.zeros(3)
    head_wind = jnp.array([-10.0, 0.0, 0.0])  # 10 m/s headwind in body-x direction
    tail_wind = jnp.array([10.0, 0.0, 0.0])   # 10 m/s tailwind

    F_calm, _, v_air_calm = compute_aircraft_forces_moments(state, actuators, wind_body=calm_wind)
    F_head, _, v_air_head = compute_aircraft_forces_moments(state, actuators, wind_body=head_wind)
    F_tail, _, v_air_tail = compute_aircraft_forces_moments(state, actuators, wind_body=tail_wind)

    assert v_air_head[0] > v_air_calm[0] > v_air_tail[0], "Airspeed ordering mismatch"
    assert F_head[0] < F_calm[0] < F_tail[0], "Drag/force ordering mismatch"
    print(
        "  Fx body [N] head/calm/tail: "
        f"{float(F_head[0]):.3f} / {float(F_calm[0]):.3f} / {float(F_tail[0]):.3f}"
    )
    print("  ✓ Steady wind effect test passed")


def test_one_minus_cosine_gust():
    """Verify one-minus-cosine gust ramp, hold, and decay behavior."""
    print("\n=== TEST: One-Minus-Cosine Gust Profile ===")
    wind_config = create_wind_config(
        enable_gust=True,
        gust_direction_ned=jnp.array([0.0, 1.0, 0.0]),
        gust_magnitude=12.0,
        gust_start_time=0.0,
        gust_rise_time=1.0,
        gust_hold_time=1.0,
    )
    turbulence = jnp.zeros(3)

    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mags = []
    for t in times:
        w = compute_total_wind_ned(t, turbulence, wind_config)
        mags.append(float(jnp.linalg.norm(w)))

    assert mags[0] < 1e-6, f"Expected near-zero gust at start, got {mags[0]}"
    assert 5.0 < mags[1] < 7.0, f"Expected ramp at t=0.5s, got {mags[1]}"
    assert abs(mags[2] - 12.0) < 1e-3, f"Expected full gust at t=1.0s, got {mags[2]}"
    assert abs(mags[3] - 12.0) < 1e-3, f"Expected held gust at t=1.5s, got {mags[3]}"
    assert abs(mags[4] - 12.0) < 1e-3, f"Expected held gust at t=2.0s, got {mags[4]}"
    assert 5.0 < mags[5] < 7.0, f"Expected decay at t=2.5s, got {mags[5]}"
    assert mags[6] < 1e-3, f"Expected near-zero gust at end, got {mags[6]}"
    print(f"  Gust magnitudes: {[round(m, 3) for m in mags]}")
    print("  ✓ Gust profile test passed")


def test_dryden_reproducibility():
    """Dryden turbulence should be deterministic for fixed seed and non-trivial."""
    print("\n=== TEST: Dryden Reproducibility ===")
    wind_config = create_wind_config(
        enable_turbulence=True,
        turbulence_sigma=jnp.array([2.0, 1.5, 0.8]),
        turbulence_length_scale=jnp.array([200.0, 200.0, 50.0]),
    )
    state = _initial_state()
    dt = 0.02
    steps = 400

    def rollout(seed: int):
        key = jax.random.PRNGKey(seed)
        turbulence = jnp.zeros(3)
        t = 0.0
        winds = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            turbulence, wind_ned = step_wind_model(
                plane_state=state,
                turbulence_ned=turbulence,
                time=t,
                key=subkey,
                dt=dt,
                wind_config=wind_config,
            )
            winds.append(np.array(wind_ned))
            t += dt
        return np.array(winds)

    wind_a = rollout(0)
    wind_b = rollout(0)
    wind_c = rollout(1)

    assert np.allclose(wind_a, wind_b), "Turbulence should be deterministic for same seed"
    assert not np.allclose(wind_a, wind_c), "Different seeds should produce different turbulence"
    assert np.std(wind_a[:, 0]) > 0.05, "Turbulence variance too small"
    print(f"  Std dev [u, v, w]: {wind_a.std(axis=0)}")
    print("  ✓ Dryden reproducibility test passed")


def test_env_wind_state():
    """Ensure environment carries and updates wind state."""
    print("\n=== TEST: Env Wind State Wiring ===")
    wind_config = create_wind_config(
        steady_wind_ned=jnp.array([3.0, -1.0, 0.0]),
        enable_turbulence=True,
        turbulence_sigma=jnp.array([0.5, 0.5, 0.2]),
    )
    env = make_env("tuned_pid_config.json", wind_config=wind_config)
    state, _ = env["reset_fn"](jax.random.PRNGKey(123))
    next_state, _, _, _, _ = env["step_fn"](state, jnp.zeros(4), jax.random.PRNGKey(456))

    assert state.wind_ned.shape == (3,), f"Expected wind shape (3,), got {state.wind_ned.shape}"
    assert next_state.turbulence_ned.shape == (3,), "Missing turbulence state"
    assert np.linalg.norm(np.array(next_state.wind_ned)) > 0.0, "Wind should be non-zero"
    print(f"  Reset wind NED: {np.array(state.wind_ned)}")
    print(f"  Step wind NED:  {np.array(next_state.wind_ned)}")
    print("  ✓ Env wind state test passed")


def main():
    print("=" * 60)
    print("Wind Model Test Suite")
    print("=" * 60)

    test_steady_wind_aero_effect()
    test_one_minus_cosine_gust()
    test_dryden_reproducibility()
    test_env_wind_state()

    print("\n" + "=" * 60)
    print("✓ ALL WIND TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

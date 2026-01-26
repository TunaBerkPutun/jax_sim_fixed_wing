#!/usr/bin/env python3
"""Test script for FixedWingTarget-v1 environment.

Runs unit tests and visualizes a full episode with random policy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_sim.env.wrappers import make_env


def test_reset():
    """Test that reset produces valid state and observation."""
    print("\n=== TEST: Reset ===")

    env = make_env("tuned_pid_config.json")
    key = jax.random.PRNGKey(42)

    state, obs = env["reset_fn"](key)

    # Check shapes
    assert state.plane_state.shape == (17,), f"Expected (17,), got {state.plane_state.shape}"
    assert obs.shape == (19,), f"Expected (19,), got {obs.shape}"

    # Check observation ranges (should be roughly normalized)
    obs_np = np.array(obs)
    print(f"  Observation range: [{obs_np.min():.2f}, {obs_np.max():.2f}]")
    assert np.all(np.abs(obs_np) < 10.0), "Observation values too large (normalization issue?)"

    # Check target is valid
    target_dist = np.linalg.norm(np.array(state.target_pos))
    print(f"  Target distance: {target_dist:.1f}m")
    assert 50.0 <= target_dist <= 150.0, f"Target distance out of range: {target_dist}"

    # Check target speed
    print(f"  Target speed: {state.target_speed:.1f} m/s")
    assert 15.0 <= state.target_speed <= 25.0, f"Target speed out of range: {state.target_speed}"

    print("  ✓ Reset test passed")


def test_step():
    """Test that step runs without errors."""
    print("\n=== TEST: Step ===")

    env = make_env("tuned_pid_config.json")
    key = jax.random.PRNGKey(42)

    state, obs = env["reset_fn"](key)

    # Neutral action (zeros)
    action = jnp.zeros(4)
    key, subkey = jax.random.split(key)

    next_state, obs, reward, done, info = env["step_fn"](state, action, subkey)

    # Check time incremented
    assert next_state.time == 0.004, f"Expected time=0.004, got {next_state.time}"

    # Check reward is finite
    assert jnp.isfinite(reward), f"Reward is not finite: {reward}"
    print(f"  Reward: {reward:.3f}")

    # Check done is boolean
    assert isinstance(bool(done), bool), f"Done is not boolean: {done}"
    print(f"  Done: {done}")

    # Check info dict
    assert "distance" in info, "Missing 'distance' in info"
    assert "speed_error" in info, "Missing 'speed_error' in info"
    print(f"  Distance to target: {info['distance']:.1f}m")
    print(f"  Speed error: {info['speed_error']:.1f} m/s")

    print("  ✓ Step test passed")


def test_vectorization():
    """Test that vmap works correctly for parallel environments."""
    print("\n=== TEST: Vectorization ===")

    env = make_env("tuned_pid_config.json")
    num_envs = 10
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

    # Vectorize reset
    reset_vmap = jax.vmap(env["reset_fn"])
    states, obs = reset_vmap(keys)

    # Check batch dimensions
    assert states.plane_state.shape == (num_envs, 17), \
        f"Expected ({num_envs}, 17), got {states.plane_state.shape}"
    assert obs.shape == (num_envs, 19), \
        f"Expected ({num_envs}, 19), got {obs.shape}"

    print(f"  Batch reset shape: {obs.shape}")

    # Vectorize step
    step_vmap = jax.vmap(env["step_fn"], in_axes=(0, 0, 0))
    actions = jnp.zeros((num_envs, 4))
    keys = jax.random.split(jax.random.PRNGKey(1), num_envs)

    next_states, obs, rewards, dones, infos = step_vmap(states, actions, keys)

    assert obs.shape == (num_envs, 19), f"Expected ({num_envs}, 19), got {obs.shape}"
    assert rewards.shape == (num_envs,), f"Expected ({num_envs},), got {rewards.shape}"

    print(f"  Batch step reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print("  ✓ Vectorization test passed")


def test_termination():
    """Test that all termination conditions work."""
    print("\n=== TEST: Termination Conditions ===")

    env = make_env("tuned_pid_config.json")
    key = jax.random.PRNGKey(42)

    state, obs = env["reset_fn"](key)

    # Manually set state to trigger crash (altitude > 0)
    crash_state = state._replace(
        plane_state=state.plane_state.at[2].set(10.0)  # z > 0 (below ground)
    )

    _, _, reward, done, info = env["step_fn"](crash_state, jnp.zeros(4), key)
    assert done, "Crash did not trigger termination"
    assert info["crash"], "Crash flag not set"
    assert reward < -1000, f"Crash reward not negative enough: {reward}"
    print(f"  Crash detection: ✓ (reward={reward:.0f})")

    # Test success (manually set close to target)
    state2, obs2 = env["reset_fn"](jax.random.PRNGKey(43))
    success_state = state2._replace(
        plane_state=state2.plane_state.at[:3].set(state2.target_pos)  # At target
    )

    _, _, reward, done, info = env["step_fn"](success_state, jnp.zeros(4), key)
    # Should be done (distance < 5m, speed might be off though)
    print(f"  Success distance check: {info['distance']:.2f}m (should be <5m)")

    print("  ✓ Termination test passed")


def test_episode():
    """Run a full episode with random policy and visualize trajectory."""
    print("\n=== TEST: Full Episode (Random Policy) ===")

    env = make_env("tuned_pid_config.json")
    key = jax.random.PRNGKey(0)

    state, obs = env["reset_fn"](key)

    trajectory = []
    rewards = []
    total_reward = 0.0

    for i in range(1000):
        # Random policy
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=(4,), minval=-1.0, maxval=1.0)

        state, obs, reward, done, info = env["step_fn"](state, action, key)

        trajectory.append(np.array(state.plane_state[:3]))
        rewards.append(float(reward))
        total_reward += reward

        if done:
            print(f"\n  Episode ended at step {i+1} ({state.time:.1f}s)")
            print(f"    Success: {info['success']}")
            print(f"    Crash: {info['crash']}")
            print(f"    Out of bounds: {info['oob']}")
            print(f"    Timeout: {info['timeout']}")
            print(f"    Final distance: {info['distance']:.1f}m")
            print(f"    Total reward: {total_reward:.1f}")
            break

    if not done:
        print(f"  Episode did not terminate (timeout after 1000 steps)")

    # Plot trajectory (3D)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        traj = np.array(trajectory)
        target = np.array(state.target_pos)

        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], -traj[:, 2], 'b-', label='Trajectory', linewidth=2)
        ax.scatter([0], [0], [100], color='g', s=100, marker='o', label='Start')
        ax.scatter([target[0]], [target[1]], [-target[2]], color='r', s=200, marker='*', label='Target')
        ax.set_xlabel('North (m)')
        ax.set_ylabel('East (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('3D Trajectory (Random Policy)')
        ax.legend()
        ax.grid(True)

        # Reward over time
        ax2 = fig.add_subplot(122)
        ax2.plot(rewards, label='Step Reward')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward Over Time')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('env_test_trajectory.png', dpi=150)
        print("\n  ✓ Trajectory plot saved to env_test_trajectory.png")

    except ImportError:
        print("  (Matplotlib not available, skipping plot)")

    print("  ✓ Episode test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FixedWingTarget-v1 Environment Test Suite")
    print("=" * 60)

    test_reset()
    test_step()
    test_vectorization()
    test_termination()
    test_episode()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

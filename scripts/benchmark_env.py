#!/usr/bin/env python3
"""Benchmark script for FixedWingTarget-v1 environment.

Measures throughput (steps/second) for different batch sizes.
"""

import time
import argparse

import jax
import jax.numpy as jnp

from jax_sim.env.wrappers import make_env
from jax_sim.physics.wind import create_wind_config


def build_wind_config_from_args(args):
    """Build wind configuration from CLI arguments."""
    return create_wind_config(
        steady_wind_ned=jnp.array([args.wind_north, args.wind_east, args.wind_down]),
        enable_gust=args.enable_gust,
        gust_direction_ned=jnp.array([args.gust_dir_north, args.gust_dir_east, args.gust_dir_down]),
        gust_magnitude=args.gust_magnitude,
        gust_start_time=args.gust_start,
        gust_rise_time=args.gust_rise,
        gust_hold_time=args.gust_hold,
        enable_turbulence=args.enable_turbulence,
        turbulence_sigma=jnp.array([args.sigma_u, args.sigma_v, args.sigma_w]),
        turbulence_length_scale=jnp.array([args.length_u, args.length_v, args.length_w]),
    )


def benchmark_throughput(num_envs: int = 4096, num_steps: int = 100, wind_config=None):
    """Measure environment throughput (steps per second).

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps to run

    Returns:
        steps_per_second: Throughput measurement
    """
    print(f"\n=== Benchmarking {num_envs} environments × {num_steps} steps ===")

    # Create environment
    env = make_env("tuned_pid_config.json", wind_config=wind_config)

    # Generate random keys for each environment
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

    # Vectorize reset and step
    reset_vmap = jax.vmap(env["reset_fn"])
    step_vmap = jax.vmap(env["step_fn"], in_axes=(0, 0, 0))

    print("  Compiling (JIT)...")
    compile_start = time.time()

    # Initialize environments
    states, obs = reset_vmap(keys)

    # Warmup: Run one step to trigger JIT compilation
    actions = jnp.zeros((num_envs, 4))
    states, obs, rewards, dones, infos = step_vmap(states, actions, keys)
    jax.block_until_ready(states)  # Wait for GPU

    compile_time = time.time() - compile_start
    print(f"  Compilation time: {compile_time:.2f}s")

    # Reset for actual benchmark
    states, obs = reset_vmap(keys)
    jax.block_until_ready(states)

    # Benchmark
    print("  Running benchmark...")
    start = time.time()

    for _ in range(num_steps):
        # Random actions
        actions = jnp.zeros((num_envs, 4))  # Could use random actions
        states, obs, rewards, dones, infos = step_vmap(states, actions, keys)

    jax.block_until_ready(states)  # Wait for all computations
    elapsed = time.time() - start

    # Calculate throughput
    total_steps = num_envs * num_steps
    steps_per_second = total_steps / elapsed

    print(f"\n  Results:")
    print(f"    Total steps: {total_steps:,}")
    print(f"    Wall time: {elapsed:.2f}s")
    print(f"    Throughput: {steps_per_second:,.0f} steps/second")
    print(f"    Per-step time: {elapsed/total_steps*1000:.3f}ms")

    return steps_per_second


def benchmark_scaling(wind_config=None):
    """Benchmark throughput at different batch sizes."""
    print("\n" + "=" * 60)
    print("FixedWingTarget-v1 Throughput Benchmark")
    print("=" * 60)

    batch_sizes = [1, 16, 64, 256, 1024, 4096]
    num_steps = 100

    results = []

    for batch_size in batch_sizes:
        try:
            sps = benchmark_throughput(
                num_envs=batch_size,
                num_steps=num_steps,
                wind_config=wind_config,
            )
            results.append((batch_size, sps))
        except Exception as e:
            print(f"  ERROR at batch_size={batch_size}: {e}")
            results.append((batch_size, 0.0))

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Batch Size':<15} {'Steps/Second':<20} {'Speedup':<10}")
    print("-" * 60)

    baseline_sps = results[0][1] if results[0][1] > 0 else 1.0

    for batch_size, sps in results:
        speedup = sps / baseline_sps if baseline_sps > 0 else 0.0
        print(f"{batch_size:<15,} {sps:<20,.0f} {speedup:<10.1f}x")

    print("=" * 60)


def benchmark_single_env(wind_config=None):
    """Benchmark single environment (no vectorization)."""
    print("\n=== Single Environment Benchmark ===")

    env = make_env("tuned_pid_config.json", wind_config=wind_config)
    key = jax.random.PRNGKey(0)

    # JIT compile
    print("  Compiling...")
    state, obs = env["reset_fn"](key)
    action = jnp.zeros(4)
    state, obs, reward, done, info = env["step_fn"](state, action, key)
    jax.block_until_ready(state)

    # Reset for benchmark
    state, obs = env["reset_fn"](jax.random.PRNGKey(1))

    num_steps = 1000
    print(f"  Running {num_steps} steps...")

    start = time.time()
    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        action = jnp.zeros(4)
        state, obs, reward, done, info = env["step_fn"](state, action, subkey)

    jax.block_until_ready(state)
    elapsed = time.time() - start

    sps = num_steps / elapsed

    print(f"  Throughput: {sps:,.0f} steps/second")
    print(f"  Per-step time: {elapsed/num_steps*1000:.3f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FixedWingTarget-v1 environment")
    parser.add_argument("--single", action="store_true", help="Benchmark single environment only")
    parser.add_argument("--batch-size", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--wind-north", type=float, default=0.0, help="Steady wind north [m/s]")
    parser.add_argument("--wind-east", type=float, default=0.0, help="Steady wind east [m/s]")
    parser.add_argument("--wind-down", type=float, default=0.0, help="Steady wind down [m/s]")
    parser.add_argument("--enable-gust", action="store_true", help="Enable one-minus-cosine gust")
    parser.add_argument("--gust-magnitude", type=float, default=0.0, help="Gust magnitude [m/s]")
    parser.add_argument("--gust-start", type=float, default=5.0, help="Gust start time [s]")
    parser.add_argument("--gust-rise", type=float, default=2.0, help="Gust rise/fall time [s]")
    parser.add_argument("--gust-hold", type=float, default=2.0, help="Gust hold time [s]")
    parser.add_argument("--gust-dir-north", type=float, default=1.0, help="Gust direction north")
    parser.add_argument("--gust-dir-east", type=float, default=0.0, help="Gust direction east")
    parser.add_argument("--gust-dir-down", type=float, default=0.0, help="Gust direction down")
    parser.add_argument("--enable-turbulence", action="store_true", help="Enable Dryden turbulence")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="Turbulence sigma-u [m/s]")
    parser.add_argument("--sigma-v", type=float, default=1.0, help="Turbulence sigma-v [m/s]")
    parser.add_argument("--sigma-w", type=float, default=0.5, help="Turbulence sigma-w [m/s]")
    parser.add_argument("--length-u", type=float, default=200.0, help="Dryden length scale Lu [m]")
    parser.add_argument("--length-v", type=float, default=200.0, help="Dryden length scale Lv [m]")
    parser.add_argument("--length-w", type=float, default=50.0, help="Dryden length scale Lw [m]")
    args = parser.parse_args()
    wind_config = build_wind_config_from_args(args)

    if args.single:
        benchmark_single_env(wind_config=wind_config)
    else:
        if args.batch_size is not None:
            benchmark_throughput(
                num_envs=args.batch_size,
                num_steps=args.num_steps,
                wind_config=wind_config,
            )
        else:
            benchmark_scaling(wind_config=wind_config)


if __name__ == "__main__":
    main()

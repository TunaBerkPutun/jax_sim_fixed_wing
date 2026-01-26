#!/usr/bin/env python3
"""Benchmark script for FixedWingTarget-v1 environment.

Measures throughput (steps/second) for different batch sizes.
"""

import time
import argparse

import jax
import jax.numpy as jnp

from jax_sim.env.wrappers import make_env


def benchmark_throughput(num_envs: int = 4096, num_steps: int = 100):
    """Measure environment throughput (steps per second).

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps to run

    Returns:
        steps_per_second: Throughput measurement
    """
    print(f"\n=== Benchmarking {num_envs} environments × {num_steps} steps ===")

    # Create environment
    env = make_env("tuned_pid_config.json")

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


def benchmark_scaling():
    """Benchmark throughput at different batch sizes."""
    print("\n" + "=" * 60)
    print("FixedWingTarget-v1 Throughput Benchmark")
    print("=" * 60)

    batch_sizes = [1, 16, 64, 256, 1024, 4096]
    num_steps = 100

    results = []

    for batch_size in batch_sizes:
        try:
            sps = benchmark_throughput(num_envs=batch_size, num_steps=num_steps)
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


def benchmark_single_env():
    """Benchmark single environment (no vectorization)."""
    print("\n=== Single Environment Benchmark ===")

    env = make_env("tuned_pid_config.json")
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
    args = parser.parse_args()

    if args.single:
        benchmark_single_env()
    else:
        if args.batch_size is not None:
            benchmark_throughput(num_envs=args.batch_size, num_steps=args.num_steps)
        else:
            benchmark_scaling()


if __name__ == "__main__":
    main()

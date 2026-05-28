#!/usr/bin/env python3
"""GPU throughput benchmark for the expert cascade with random waypoints.

For each batch size, generates N random NED waypoints in a reasonable box,
vmaps the cascade + dynamics, runs a 60s rollout, and measures wall-clock
steps/second. Reports per-env steps/sec and total steps/sec so scaling is
visible at a glance.

Usage:
    uv run python scripts/benchmark_expert.py
    uv run python scripts/benchmark_expert.py --batch-sizes 1,32,256,1024
    uv run python scripts/benchmark_expert.py --duration 30 --reach-threshold 30
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from jax import random

from jax_sim.controllers.fixed_wing.expert import (
    default_expert_config,
    expert_goto_step,
    init_expert_state,
    load_expert_config,
)
from jax_sim.vehicles.fixed_wing.tier1 import equations_of_motion


def _initial_plant_state() -> jnp.ndarray:
    """Level cruise at 100 m, 20 m/s North."""
    return jnp.array([
        0.0, 0.0, -100.0,
        20.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.5,
    ])


def sample_random_targets(
    key: jax.Array,
    n: int,
    xy_range: float = 150.0,
    altitude_min: float = 80.0,
    altitude_max: float = 120.0,
    min_dist: float = 50.0,
) -> jnp.ndarray:
    """Uniform random targets in a NED box, with a minimum XY-distance floor."""
    k_xy, k_alt = random.split(key)
    # Sample XY in [-xy_range, +xy_range], reject inside a min_dist disk by scaling.
    xy = random.uniform(k_xy, (n, 2), minval=-xy_range, maxval=xy_range)
    norms = jnp.linalg.norm(xy, axis=1, keepdims=True)
    # Smoothly push points outward from the origin so |xy| >= min_dist.
    scale = jnp.maximum(min_dist / jnp.maximum(norms, 1e-6), 1.0)
    xy = xy * scale
    # Altitude (NED z = -alt).
    alt = random.uniform(k_alt, (n, 1), minval=altitude_min, maxval=altitude_max)
    z = -alt
    return jnp.concatenate([xy, z], axis=1)  # (n, 3)


def make_rollout(config, dt: float, n_steps: int):
    """Build a jit'd rollout function for one (state, target) pair.

    Memory-efficient: carries `closest_so_far` inside the scan instead of
    materializing the per-step trajectory. Memory is O(batch · state_dim)
    rather than O(batch · n_steps · state_dim), which is what lets us scale
    to batch sizes in the millions.
    """
    wind = jnp.zeros(3)
    airspeed_sp = jnp.asarray(20.0)

    @jax.jit
    def rollout(plant_state, target):
        es = init_expert_state(plant_state, target)
        closest0 = jnp.linalg.norm(plant_state[0:2] - target[0:2])

        def step(carry, _):
            ps, es_, closest = carry
            act, es_new, _ = expert_goto_step(
                ps, target, airspeed_sp, es_, config, dt, wind
            )
            ps_new = equations_of_motion(ps, act, dt)
            xy_err = jnp.linalg.norm(ps_new[0:2] - target[0:2])
            return (ps_new, es_new, jnp.minimum(closest, xy_err)), None

        (ps_final, _, closest_xy), _ = jax.lax.scan(
            step, (plant_state, es, closest0), jnp.arange(n_steps)
        )
        final_alt = -ps_final[2]
        final_speed = jnp.linalg.norm(ps_final[3:6])
        return closest_xy, final_alt, final_speed

    return jax.vmap(rollout, in_axes=(None, 0))


def benchmark_batch(config, batch_size: int, duration: float, dt: float,
                    reach_threshold: float, seed: int) -> dict:
    n_steps = int(duration / dt)
    rollout_v = make_rollout(config, dt, n_steps)

    key = random.PRNGKey(seed)
    targets = sample_random_targets(key, batch_size)
    plant0 = _initial_plant_state()

    # Warm-up — first call compiles + initial CUDA buffer allocation.
    closest, alt, spd = rollout_v(plant0, targets)
    jax.block_until_ready(closest)

    # Timed pass.
    t0 = time.perf_counter()
    closest, alt, spd = rollout_v(plant0, targets)
    jax.block_until_ready(closest)
    wall = time.perf_counter() - t0

    total_steps = batch_size * n_steps
    steps_per_sec = total_steps / wall
    per_env_steps_per_sec = n_steps / wall
    reach_rate = float(jnp.mean(closest < reach_threshold))

    return {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "wall_s": wall,
        "total_steps_per_s": steps_per_sec,
        "per_env_steps_per_s": per_env_steps_per_sec,
        "reach_rate": reach_rate,
        "closest_p50_m": float(jnp.median(closest)),
        "closest_p90_m": float(jnp.percentile(closest, 90)),
        "alt_p50": float(jnp.median(alt)),
        "spd_p50": float(jnp.median(spd)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,16,64,256,1024,4096",
                        help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Rollout duration per env [s].")
    parser.add_argument("--dt", type=float, default=0.004,
                        help="Sim timestep [s] (250 Hz default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach-threshold", type=float, default=30.0,
                        help="Closest-XY threshold (m) for reach-rate metric.")
    parser.add_argument("--config", default="tuned_expert_config.json",
                        help="ExpertConfig JSON (falls back to default if absent).")
    args = parser.parse_args()

    try:
        config = load_expert_config(args.config)
        print(f"Loaded config from {args.config}")
    except FileNotFoundError:
        config = default_expert_config()
        print(f"{args.config} not found — using default_expert_config()")

    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"Rollout: {args.duration:.0f}s × {int(args.duration/args.dt)} steps @ {1/args.dt:.0f} Hz")
    print(f"Reach threshold: {args.reach_threshold:.0f} m")
    print()

    header = (
        f"{'batch':>6} | {'wall (s)':>8} | "
        f"{'steps/s total':>14} | {'steps/s/env':>12} | "
        f"{'reach@thr':>9} | {'closest p50/p90':>16} | {'final alt/V':>13}"
    )
    print(header); print("-" * len(header))

    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        try:
            r = benchmark_batch(config, bs, args.duration, args.dt,
                                args.reach_threshold, args.seed)
            print(f"{r['batch_size']:>6} | {r['wall_s']:>8.2f} | "
                  f"{r['total_steps_per_s']:>14,.0f} | "
                  f"{r['per_env_steps_per_s']:>12,.0f} | "
                  f"{r['reach_rate']*100:>8.1f}% | "
                  f"{r['closest_p50_m']:>6.1f}/{r['closest_p90_m']:>7.1f} m | "
                  f"{r['alt_p50']:>5.0f} m/{r['spd_p50']:>5.1f}")
        except Exception as e:
            print(f"{bs:>6} | FAILED: {type(e).__name__}: {e}")
            break


if __name__ == "__main__":
    main()

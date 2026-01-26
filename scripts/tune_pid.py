#!/usr/bin/env python3
"""Auto-tune PID gains using Evolution Strategy.

Uses JAX's vmap to evaluate thousands of PID configs in parallel.

Usage:
    uv run python scripts/tune_pid.py
    uv run python scripts/tune_pid.py --pop-size 4096 --generations 100
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from jax_sim.controllers.tuning import run_es_tuning, TuningResult
from jax_sim.controllers.tuning.es_tuner import save_tuned_config


def plot_tuning_history(result: TuningResult, save_path: str = "tuning_history.png"):
    """Plot the loss history during tuning."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.loss_history, "b-", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Loss")
    ax.set_title("ES Tuning Progress")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Mark final loss
    ax.axhline(result.final_loss, color="r", linestyle="--", alpha=0.7,
               label=f"Final: {result.final_loss:.1f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved tuning history to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-tune PID gains with ES")
    parser.add_argument("--pop-size", type=int, default=2048,
                        help="Population size (default: 2048)")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations (default: 50)")
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Mutation noise (default: 0.15)")
    parser.add_argument("--lr", type=float, default=0.2,
                        help="Learning rate (default: 0.2)")
    parser.add_argument("--elite-ratio", type=float, default=0.1,
                        help="Elite ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--stage", type=str, default="rate",
                        choices=["rate", "full"],
                        help="Tuning stage: rate (default) or full cascade")
    parser.add_argument("--method", type=str, default="es",
                        choices=["es", "model"],
                        help="Tuning method: es (default) or model-based")
    parser.add_argument("--rate-wn", type=float, default=8.0,
                        help="Model-based rate tuning natural frequency (rad/s)")
    parser.add_argument("--rate-zeta", type=float, default=0.7,
                        help="Model-based rate tuning damping ratio")
    parser.add_argument("--rate-kp-target", type=float, default=0.2,
                        help="Model-based rate tuning target Kp floor")
    parser.add_argument("--rate-kp-margin", type=float, default=0.05,
                        help="Extra Kp margin to ensure stable linear term")
    parser.add_argument("--debug-nonfinite", action="store_true",
                        help="Log CSVs for non-finite loss candidates")
    parser.add_argument("--debug-output-dir", type=str, default="pid_debug",
                        help="Directory for debug CSV logs")
    parser.add_argument("--debug-log-limit", type=int, default=5,
                        help="Max number of debug CSVs to write")
    parser.add_argument("--output", type=str, default="tuned_pid_config.json",
                        help="Output config file (default: tuned_pid_config.json)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    print("=" * 60)
    print("PID AUTO-TUNER (Evolution Strategy)")
    print("=" * 60)
    print(f"Population: {args.pop_size}")
    print(f"Generations: {args.generations}")
    print(f"σ = {args.sigma}, lr = {args.lr}, elite = {args.elite_ratio:.0%}")
    print(f"Stage: {args.stage}")
    print("=" * 60)

    start_time = time.time()

    if args.method == "model":
        if args.stage != "rate":
            raise SystemExit("Model-based tuning currently supports only --stage rate.")
        from jax_sim.controllers.tuning.model_tuner import run_model_tuning_rate
        result = run_model_tuning_rate(
            throttle_cmds=(0.2, 0.9),
            wn=args.rate_wn,
            zeta=args.rate_zeta,
            kp_target=args.rate_kp_target,
            kp_margin=args.rate_kp_margin,
            seed=args.seed,
            verbose=True,
        )
    else:
        result = run_es_tuning(
            pop_size=args.pop_size,
            generations=args.generations,
            sigma=args.sigma,
            learning_rate=args.lr,
            elite_ratio=args.elite_ratio,
            seed=args.seed,
            verbose=True,
            tune_rate_only=(args.stage == "rate"),
            debug_nonfinite=args.debug_nonfinite,
            debug_output_dir=args.debug_output_dir,
            debug_log_limit=args.debug_log_limit,
        )

    elapsed = time.time() - start_time
    print(f"\nTuning completed in {elapsed:.1f}s")

    # Save config
    save_tuned_config(result, args.output)

    # Plot history
    if not args.no_plot:
        plot_tuning_history(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune the expert cascade and write `tuned_expert_config.json`.

Usage:
    uv run python scripts/tune_expert.py
    uv run python scripts/tune_expert.py --rate-wn 5.0 --attitude-ratio 4.0
"""

from __future__ import annotations

import argparse
import time

from jax_sim.controllers.fixed_wing.expert import save_expert_config
from jax_sim.controllers.tuning.expert_tuner import tune_expert


def main():
    parser = argparse.ArgumentParser(description="Tune the expert cascade.")
    parser.add_argument("--rate-wn", type=float, default=5.0,
                        help="Rate-loop natural frequency [rad/s] (PX4 target ~5).")
    parser.add_argument("--rate-zeta", type=float, default=0.7,
                        help="Rate-loop damping ratio.")
    parser.add_argument("--rate-ff-boost", type=float, default=1.0,
                        help="Multiplier on rate FF above analytical -a/b. "
                             "Default 1.0 gives the analytically-correct FF "
                             "that yields zero steady-state error and matches "
                             "the closed-loop bandwidth set by --rate-wn. "
                             "Bump for more transient aggressiveness (PX4 yaml "
                             "FW_RR_FF=0.5 corresponds to ~5× for the SIH "
                             "airframe); past ~3× the rate loop starts to "
                             "overshoot and BPTT gradients explode faster.")
    parser.add_argument("--attitude-ratio", type=float, default=4.0,
                        help="Outer/inner bandwidth separation (τ = ratio/wn).")
    parser.add_argument("--airspeed-trim", type=float, default=20.0,
                        help="Trim airspeed [m/s].")
    parser.add_argument("--throttle-trim", type=float, default=0.6,
                        help="Trim throttle [0..1].")
    parser.add_argument("--altitude-pole", type=float, default=0.5,
                        help="TECS altitude closed-loop pole [rad/s].")
    parser.add_argument("--energy-pole", type=float, default=1.0,
                        help="TECS energy-rate closed-loop pole [rad/s].")
    parser.add_argument("--dt", type=float, default=0.004,
                        help="Timestep [s].")
    parser.add_argument("--output", type=str, default="tuned_expert_config.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERT CASCADE TUNER (jacrev + pole placement)")
    print("=" * 60)
    print(f"  rate_wn         = {args.rate_wn}")
    print(f"  rate_zeta       = {args.rate_zeta}")
    print(f"  attitude_ratio  = {args.attitude_ratio} → τ = {args.attitude_ratio/args.rate_wn:.3f} s")
    print(f"  airspeed_trim   = {args.airspeed_trim} m/s")
    print(f"  altitude_pole   = {args.altitude_pole} rad/s")
    print(f"  energy_pole     = {args.energy_pole} rad/s")
    print("-" * 60)

    t0 = time.time()
    cfg = tune_expert(
        throttle_cmds=(0.3, 0.7),
        rate_wn=args.rate_wn,
        rate_zeta=args.rate_zeta,
        rate_ff_boost=args.rate_ff_boost,
        attitude_ratio=args.attitude_ratio,
        airspeed_trim=args.airspeed_trim,
        throttle_trim=args.throttle_trim,
        altitude_pole=args.altitude_pole,
        energy_pole=args.energy_pole,
        dt=args.dt,
    )
    dt_run = time.time() - t0
    print(f"Tuned in {dt_run:.1f}s.")
    print()
    print("Rate loop:")
    print(f"  rate_kp = {[round(float(x), 4) for x in cfg.rate_kp]}")
    print(f"  rate_ki = {[round(float(x), 4) for x in cfg.rate_ki]}")
    print(f"  rate_ff = {[round(float(x), 4) for x in cfg.rate_ff]}")
    print("Attitude:")
    print(f"  tau_roll  = {float(cfg.tau_roll):.3f}")
    print(f"  tau_pitch = {float(cfg.tau_pitch):.3f}")
    print("TECS:")
    print(f"  alt_p     = {float(cfg.tecs_alt_p):.4f}")
    print(f"  thr_p, thr_i = {float(cfg.tecs_thr_p):.5f}, {float(cfg.tecs_thr_i):.5f}")
    print(f"  pit_p, pit_i = {float(cfg.tecs_pitch_p):.5f}, {float(cfg.tecs_pitch_i):.5f}")
    print("NPFG:")
    print(f"  period  = {float(cfg.npfg_period):.3f}, damping = {float(cfg.npfg_damping):.3f}")

    save_expert_config(cfg, args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

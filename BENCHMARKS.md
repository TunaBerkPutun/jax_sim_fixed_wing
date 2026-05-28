# Benchmarks

Throughput and quality baselines for the expert cascade. Update this file
when you change anything in the inner loop (aero, dynamics, controller) so
we have a directly comparable A/B.

## Baseline — `expert` cascade, default aero

**Captured:** 2026-05-12
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 590.48.01
**Software:** JAX 0.8.2, Python 3.12, `jax[cuda12]`
**Aircraft:** `DEFAULT_AIRCRAFT` (PX4 SIH defaults; 1 kg, 86 cm span)
**Config:** `tuned_expert_config.json` from
`uv run python scripts/tune_expert.py` (default flags — analytical
`rate_ff_boost=1.0`, `rate_wn=5`, `attitude_ratio=4`, etc.)
**Sim:** `DT = 0.004 s` (250 Hz), RK4 dynamics, full PX4-equivalent cascade
forward pass (NPFG → wind triangle → heading→roll → TECS → attitude → rate),
no wind.

### Throughput sweep (10 s rollouts; trajectory not materialized — closest-XY carried in the scan)

| Batch | Wall (s) | Steps/s total | Steps/s/env |
|---:|---:|---:|---:|
| 4 096 | 0.54 | 19.0 M | 4 630 |
| 8 192 | 0.58 | 35.2 M | 4 296 |
| 16 384 | 0.75 | 54.3 M | 3 316 |
| 24 576 | 0.81 | 75.4 M | 3 069 |
| **32 768** | **0.97** | **84.3 M ← peak** | 2 572 |
| 40 960 | 1.59 | 64.5 M | 1 575 |
| 49 152 | 1.66 | 73.8 M | 1 502 |
| 65 536 | 2.22 | 73.7 M | 1 125 |
| 131 072 | 4.89 | 67.0 M | 511 |
| 262 144 | 12.31 | 53.2 M | 203 |
| 524 288 | 16.67 | 39.3 M | 75 |

### Saturation summary

- **Saturation point:** batch ≈ 32 768
- **Peak throughput:** ~82–84 M sim-steps / s
- **Beyond peak:** aggregate throughput plateaus to ~64–74 M, then walks
  *backwards* past 131 k (kernels too big; memory traffic dominates).
- **Per-env real-time factor at peak:** 2 572 steps/s × 4 ms/step ≈ 10×
- **Aggregate real-time factor at peak:** 82 M × 4 ms ≈ **328 000×** real time
  (aircraft-seconds simulated per wall-second across the batch)

### Recommended comparison point for future changes

Lock benchmarks to **batch 32 768, duration 10 s, no wind** — the saturation
peak. At smaller batches the GPU is under-utilized so compute-cost changes
get masked; at larger batches you're memory-bound and noise dominates.

```bash
uv run python scripts/benchmark_expert.py --batch-sizes 32768 --duration 10
```

**Result to beat / track:**

```
batch  wall(s)   steps/s total   steps/s/env
32768    0.97        84.3 M          2 572
```

When evaluating a change:
- Wall time drops at batch 32k → throughput win on the hot path.
- Wall time rises at batch 32k but the saturation knee moves *right* →
  smaller per-step cost lets you fit more parallel envs.
- Sweep `--batch-sizes 4096,16384,32768,65536,131072` before and after to
  see the full curve, not just a single point.

## Quality — expert cascade, 1024 random waypoints

Random waypoints uniformly distributed in a `±150 m` NE box with a min-XY
distance of 50 m and altitude in `[80, 120] m`. 60 s rollouts each.

| Closest-XY metric | Value |
|---|---|
| p10 | 3.4 m |
| p50 | 18.3 m |
| p90 | 37.5 m |
| p99 | 42.8 m |
| Reach within 10 m | 27.8 % |
| Reach within 30 m | 76.8 % |
| Reach within 50 m | **100 %** |

Note the p90 of ~38 m is the U-turn worst case: when the target starts
behind the aircraft, NPFG (segment-only, no curvature) carves a wide arc
and the closest approach happens at the arc apex. NPFG-lite intentionally
ships without arrival logic — adding an arrival radius + segment-advance
manager would close that gap.

## Notes

- The throughput rollout discards per-step trajectory and folds
  `closest_so_far` into the scan carry; that's what makes batch sizes in
  the hundreds of thousands fit in 8 GB of VRAM.
- All measurements use the second pass (post-JIT-warmup); first call is
  discarded.
- The benchmark script lives at `scripts/benchmark_expert.py` and accepts
  `--batch-sizes`, `--duration`, `--dt`, `--reach-threshold`, `--config`.

---

## Aero validation — JSBSim suite

Reference comparison against JSBSim (`px4_sih_uav`, matched airframe) used
to detect aero divergence when `physics/aerodynamics.py` or
`physics/dynamics.py` are edited. Open-loop, no controller; each test case
is a **separate** rollout from the same matched trim with a per-test
duration sized so neither sim hits the ground.

> **Why independent tests, not one long batch.** A previous chained-180 s
> schedule drove both sims into a graveyard spiral during a sustained-
> aileron segment around t ≈ 86 s. JSBSim's integrator went numerically
> berserk on ground impact (q-peak of **11 286 °/s**), and every later
> "doublet" and "chirp" metric ran on garbage state. Independent rollouts
> bound the divergence per maneuver and add a ground-guard
> (`alt < 5 m → truncate and flag crashed=True`) so partial trajectories
> still produce honest metrics.

### Baseline — `aero_baseline/`, default aero, 2026-05-12 (rev 2)

- **Aircraft:** `px4_sih_uav` matched JSBSim model + `DEFAULT_AIRCRAFT` jax_sim aero.
- **Trim:** V = 20 m/s, alt = 100 m, α = −0.843°,
  `(ail, ele, rud, thr) = (0, 0.0758, 0, 0.3792)`. Same trim injected into
  both sims; JSBSim lag filters pre-seeded, jax_sim actuators pre-seeded.
- **Sample rate:** 250 Hz (dt = 0.004 s).
- **Ground guard:** truncate when altitude drops below 5 m AGL and mark
  the case `crashed=True`. Metrics are computed on the overlap window
  between the two sims.

> **Rev 2 (this baseline):** `update_actuators` now applies first-order
> lag to aileron / elevator / rudder using `tau_servo = 0.1 s` from
> `constants.py` (previously only throttle was lagged; surfaces snapped
> to commanded value in one step). The rev 1 baseline had pitch/yaw
> doublet ratios of ≈160 % and ≈143 %; rev 2 has them at ≈66 % and ≈67 %.
> The residual gap reflects a structural difference between segment-based
> jax_sim aero and JSBSim's coefficient-buildup model and is *not*
> something to tune away. See "Reading the table" below.

### Suite

| Case               | Dur [s] | Maneuver                                          | Probes                |
|---|---:|---|---|
| `trim_hold`        | 30      | hold trim                                         | drift / numerical stability |
| `elev_doublet_sm`  | 25      | elevator ±0.05, 2× 0.5 s pulses; 23 s free decay  | phugoid period / damping     |
| `elev_doublet_lg`  | 15      | elevator ±0.15, 2× 0.5 s pulses                   | short-period peak / Cmq      |
| `ail_doublet`      | 15      | aileron ±0.30, 2× 0.5 s pulses                    | roll mode / Cℓp, Cℓδa        |
| `rud_doublet`      | 15      | rudder ±0.20, 2× 0.5 s pulses                     | dutch-roll period / damping  |
| `ail_step_small`   | 8       | sustained aileron +0.03 from t=1 s                | spiral-mode buildup          |
| `throttle_step`    | 20      | throttle +0.20 step from t=1 s                    | long-period speed/alt, propulsion |
| `elev_chirp`       | 30      | elevator chirp 0.2 → 2.0 Hz, A=0.05 over 28 s     | broadband |Q/E| at 0.5/1/1.5 Hz |

### Per-case RMS divergence (JSBSim − jax_sim, default aero, rev 2)

| Case             | dh [m] | dV [m/s] | dθ [deg] | dφ [deg] | dq [°/s] | crash? |
|---|---:|---:|---:|---:|---:|---|
| trim_hold        | 0.50   | 0.10     | 0.15     | 0.00     | 0.51     | —      |
| elev_doublet_sm  | 0.39   | 0.10     | 0.17     | 0.00     | 0.69     | —      |
| elev_doublet_lg  | 0.24   | 0.12     | 0.31     | 0.00     | 1.74     | —      |
| ail_doublet      | 0.91   | 0.13     | 0.47     | 1.60     | 0.85     | —      |
| rud_doublet      | 0.20   | 0.10     | 0.15     | 0.15     | 0.73     | —      |
| ail_step_small   | 0.33   | 0.30     | 0.41     | 0.39     | 1.16     | —      |
| throttle_step    | **12.91** | 0.28  | **4.26** | 0.04     | 0.87     | —      |
| elev_chirp       | 0.34   | 0.11     | 0.26     | 0.00     | 1.38     | —      |

### Peak-response ratios jax_sim / JSBSim (canonical aero A/B targets)

| Case             | Peak (JSB / jax)       | Ratio  |
|---|---|---:|
| elev_doublet_sm  | q 11.9 / 7.8 deg/s     | **66 %** |
| elev_doublet_lg  | q 37.6 / 23.3 deg/s    | **62 %** |
| ail_doublet      | p 168.4 / 159.3 deg/s  | **95 %** |
| rud_doublet      | r 63.1 / 42.2 deg/s    | **67 %** |
| throttle_step    | dh 117 / 93 m          | **79 %** |
| elev_chirp       | |Q/E|@1Hz 1.97 / 1.70  | **87 %** |
| ail_step_small   | final_phi 70.7 / 71.3 deg | **101 %** |

Full per-state stats (mean/std/min/max/abs_max for h, V, θ, φ, ψ, p, q, r)
and per-case scalar features (peak responses, phugoid period, log-decrement
damping, chirp gains at 0.5/1.0/1.5 Hz, etc.) are in `metrics.json`.

### Reading the table

- **Static aero matches to within 1 %.** A direct moment-vs-deflection probe
  (`scripts/diagnose_aero_divergence.py`) confirms `Cmde`, `Cndr`, `Clda`
  in jax_sim line up with the JSBSim XML at all deflections −0.30 … +0.30.
  The visible transient gaps are *not* in the aero derivatives.
- **Free-response divergence is sub-metre / sub-degree** across `trim_hold`,
  `elev_doublet_*`, `ail_doublet`, `rud_doublet`, `ail_step_small`. The
  airframe stability matches.
- **Peak-response ratios sit at ~65 % on pitch and yaw, ~95 % on roll.**
  After the servo-lag fix, jax_sim reaches lower peak rates than JSBSim
  on the pitch and yaw doublets. Roll matches because the 500 ms pulse is
  much longer than the roll mode time constant (~57 ms), so both sims hit
  the quasi-steady roll rate regardless of small differences in transient
  shaping. Pitch (short-period τ ≈ 35 ms) and yaw (dutch-roll comparable)
  are faster than the pulse, so transient-stage differences dominate the
  peak. This residual reflects a **structural** difference between
  segment-based jax_sim aero and JSBSim's coefficient-buildup linear model
  and is not something to tune away.
- **`throttle_step` is the slipstream probe.** Under +0.20 throttle JSBSim
  climbs +117 m, jax_sim climbs +93 m (79 % ratio). Thrust force is
  identical in both sims; the gap is propeller slipstream amplifying
  dynamic pressure at the tail in jax_sim (no slipstream model exists in
  the JSBSim XML — every aero function multiplies by freestream `qbar`).
  Expect this number to move when slipstream is edited.
- **`ail_step_small`** is intentionally non-trivial (no controller compensates
  the bank-induced descent). It probes spiral-mode behaviour and lateral-
  longitudinal coupling. Final-roll match (101 %) is excellent — the
  spiral develops the same way in both sims.
- **`elev_chirp` |Q/E|@1Hz** is 87 % match — broadband frequency-response
  agreement is much closer than the doublet peaks because the chirp drives
  the elevator at ≤ 2 Hz, which is slow relative to the 100 ms servo lag
  (the lag attenuates equally in both sims).

### Reproduce / A/B

```bash
# Baseline already saved to aero_baseline/ — do NOT overwrite it.
# After an aero edit:
uv run python scripts/compare_jsbsim_long.py --output-dir aero_v2

# Optional: run a subset while iterating
uv run python scripts/compare_jsbsim_long.py \
    --output-dir aero_v2 --cases elev_doublet_lg,ail_doublet
```

Then diff `aero_baseline/metrics.json` ↔ `aero_v2/metrics.json` — the
`cases.<name>.divergence` block is the canonical scalar A/B target.

### Saved artifacts (do not overwrite)

```
aero_baseline/
├── metrics.json                       per-case scalars + divergence
├── summary.txt                        human-readable one-liner per case
└── <case>/
    ├── jsbsim.csv      JSBSim trajectory      (one row per outer step)
    ├── jax_sim.csv     jax_sim trajectory
    └── compare.png     8-panel overlay
```

`metrics.json` contains, per case: jsbsim & jax_sim state stats, case-
specific features (peaks, periods, gains), and the cross-sim RMS divergence
— enough to recompute any derived metric offline without re-running JSBSim.

### Existing pulse-style comparison (kept as fast smoke check)

`scripts/compare_jsbsim.py` is the **15 s single-elevator-pulse** quick test.
Use it for fast iteration during an aero edit; switch to the suite above
when you're ready to lock in a result. The two are complementary, not
redundant.

```bash
uv run python scripts/compare_jsbsim.py                 # 15 s pulse
```

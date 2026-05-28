# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

JAX-based 6-DOF fixed-wing UAV simulator. Two controllers ship in-repo:
1. **`controllers/cascade_pid`** — simple cascade PID baseline (legacy).
2. **`controllers/expert`** — full PX4-equivalent native JAX cascade
   (NPFG lateral guidance → TECS longitudinal energy → attitude P + coord-turn
   FF → rate P+I+FF + nonlinear anti-windup). Drop-in replacement target for
   future LQR / MPC / learned controllers via `controllers/base.py`.

PPO training loop is present but not the primary deliverable. Roadmap: stable
sim → PX4 SITL/HITL bridge → RL training. Python ≥ 3.12, `jax[cuda12]`,
`flax` (NNX, not Linen). Dependencies managed by `uv` — always invoke entry
points via `uv run`.

## Common Commands

```bash
# Open-loop fixed-wing sim (writes sim_log.csv, sim_summary.png)
uv run python scripts/run_sim.py

# PPO training (writes runs/<name>/ TB logs, checkpoints/<name>/*.pkl)
uv run python scripts/train_ppo.py --num-envs 32 --learning-rate 1e-4

# ES tuner for legacy PID (default: rate-only, 9 params)
uv run python scripts/tune_pid.py --pop-size 4096 --generations 100

# Expert cascade — tune (jacrev pole placement) and validate
uv run python scripts/tune_expert.py                    # writes tuned_expert_config.json
uv run python scripts/test_expert.py --mode waypoint    # 0,0,-100 → -100,-100,-100 litmus
uv run python scripts/test_expert.py --mode rate-step --amp 0.5
uv run python scripts/test_expert.py --mode attitude-step --amp 5
uv run python scripts/test_expert.py --mode grad-check  # single-step ∂(actuators)/∂(config)

# Sanity checks
uv run python scripts/test_env.py           # 5 env unit tests + random-policy rollout
uv run python scripts/test_pid.py --roll-cmd 10 --duration 5
uv run python scripts/test_wind.py          # steady / gust / Dryden / env wiring
uv run python scripts/benchmark_env.py      # throughput at batch sizes 1..4096
```

No pytest, no formatter/linter. Validation is the `scripts/test_*.py` set.

## Differentiability — the central design property

**Every public function in `physics/`, `controllers/`, and `env/` is `@jax.jit` over pure JAX ops** (no Python branches on tracers, no data-dependent shapes). The full stack `obs → action → cascade_pid → RK4 dynamics → reward` is `jax.grad`-able. This is the project's killer feature; SOTA tuning approaches that need differentiable simulators (differentiable-MPC, analytical PID synthesis via Jacobian linearization, gradient-descent on gains) are directly applicable.

Known gradient-blockers / hazards:
- `dynamics.py` ground-contact reset: `jax.lax.select(landed, zeroed_state, next_state)` with `landed = pos_z >= 0`. Gradients vanish on the crash branch — train-on-crash dynamics won't work; use soft penalties in the reward instead.
- `aerodynamics.py` stall blending uses `tanh` with a steep coefficient near ±20° AoA — smooth but ill-conditioned in that band.
- Propeller slipstream uses `sqrt(2T/(ρπr²))`; clamp protects negative thrust but the derivative is unbounded as T → 0⁺.
- Quaternion normalization clamps the norm at `1e-8` (`rigid_body.py`) — fine in practice.

## Architecture

```
RL agent ──action──► env.step ──setpoints──► cascade_pid ──actuators──► equations_of_motion
                       │                          │                          │
                       └─ EnvState (NamedTuple)   └─ PIDState/PIDConfig      └─ AircraftParams,
                          target, wind state,                                   wind_body
                          PID state, key
```

### Physics (`jax_sim/physics/`)

- **State vector**: flat `(17,) = [pos(3), vel(3), quat(4 w-x-y-z), omega(3), actuators(4)]`. NED frame; altitude is `-pos[2]`. Quaternion is body→earth.
- `equations_of_motion(state, user_cmd, dt, aircraft, wind_body)` is the canonical single step. `user_cmd ∈ [-1,1]⁴` (throttle in `[0,1]`).
- `rigid_body.rigid_body_step` is **RK4** over `[pos, vel, quat, omega]`. Forces/moments are evaluated **once per outer step** and held fixed across the four substeps (cheap and standard, but a quality knob if you later need higher fidelity). Quaternion renormalized post-integration; body rates clipped to `±max_body_rate` (6π rad/s).
- `aerodynamics.compute_fixed_wing_aero` is a **PX4-SIH-style segment model**: five segments (wing L/R with -4° incidence + dihedral, tailplane, vertical fin, fuselage). Tail and fin see propeller slipstream (`sqrt(2T/(ρπr²))`). Aileron/elevator/rudder enter **normalized [-1,1]**; `dynamics.py` divides radians by `flap_max` before calling. **Elevator sign-flip lives in `cascade_pid.py:111` (not in aero)** — the comment in `dynamics.py` is misleading; do not double-negate.
- `aircraft.AircraftParams` (`flax.struct.dataclass`) bundles `EnvironmentParams`, `MassProps`, `ActuatorParams`, `PropulsionParams`, `AeroSegments`. Single `DEFAULT_AIRCRAFT` factory; swap by passing a different `AircraftParams` rather than editing constants.
- `simulator.simulate` / `simulate_batch` wrap `equations_of_motion` in `jax.lax.scan` (+ `vmap`) for rollouts.
- `wind.py` — steady NED bias + 1-cosine gust + per-axis Dryden turbulence (OU process). `step_wind_model` advances state; `wind_ned_to_body` rotates for `equations_of_motion(wind_body=...)`.

### Controllers (`jax_sim/controllers/`)

Two controllers live alongside each other behind a common duck-typed protocol
defined in `controllers/base.py`. Any controller (PID, expert, LQR, MPC,
learned) exposes the same `(init_state, step)` signature:

```python
init_state(initial_plant_state, target, config, **kwargs) -> ControllerState
step(plant_state, target, ctrl_state, config, dt, wind_ned)
    -> (actuators, new_ctrl_state, debug)
```

The expert subpackage is the canonical reference. The simple cascade PID
stays for backward compatibility with the legacy RL env (`fixed_wing_target`).

#### Simple cascade PID — `controllers/cascade_pid.py` (legacy)

- `cascade_pid.cascade_pid_step` composes three loops:
  - **Attitude (outer, pure-P)** `attitude/pid.py` — `p_sp = (roll_cmd - roll) / tau_roll`, clipped to `rate_limit`. No rate feed-forward, no coordinated-turn rudder.
  - **Rate (inner, full PID)** `rate/pid.py` — derivative on measurement (no setpoint kick), integral clamp anti-windup.
  - **Speed (parallel PI+FF)** `speed/pid.py` — `throttle_ff` baseline + PI correction.
  - Yaw rate command bypasses the outer loop (clipped and fed directly to the rate controller).
- `PIDConfig` / `PIDState` are JAX `NamedTuple`s. Loaded from `tuned_pid_config.json` via `env.wrappers.load_tuned_pid_config`.

#### Expert cascade — `controllers/expert/` (canonical)

Native JAX 1-to-1 port of PX4's fixed-wing controller stack. Lives entirely
in this repo (no PX4 binary at runtime); the API surface mirrors PX4's so a
future MAVLink HIL bridge produces a direct comparison rather than an
architectural diff.

**Module layout** (`controllers/expert/`):
- `types.py` — `ExpertConfig`, `ExpertState`, `ExpertDebug` NamedTuples + JSON I/O.
- `rate.py` — port of `src/lib/rate_control/rate_control.cpp:71-118`.
  P + I + FF with PX4's nonlinear anti-windup (`i_factor`,
  saturation-aware error clamping, landed-gate) expressed as `jnp.where` —
  no Python branches, fully jit/grad/vmap-friendly.
- `attitude.py` — port of `FixedwingAttitudeControl.cpp:301-323`. Tau-based
  P + coordinated-turn FF (`yawrate_ff = g·sin(roll)·cos(pitch)/V`) with a
  smooth tilt-gate. **Gate input is `1 - cos(tilt)` instead of `arccos(cos_tilt)`**
  — the latter has a singular derivative at level flight which produces
  NaN gradients during backprop.
- `tecs.py` — TECS-lite (total energy + balance, two PIs). Drops PX4's
  underspeed protection (gradient cliff) and density compensation; keeps
  speed-weighting and STE/SEB low-pass filters.
- `npfg.py` — straight-segment NPFG (`DirectionalGuidance.cpp`).
  Adaptive lookahead `L = period · V_g / (2π · damping)`.
- `wind_triangle.py` — `CourseToAirspeedRefMapper` port. Bumps `airspeed_ref`
  when crosswind would make the course infeasible.
- `slew_rate.py`, `airspeed_scaling.py` — small helpers; PX4 scales torque
  by `(V_trim/V)²` to keep closed-loop bandwidth flat across airspeed.
- `step.py` — composition: `expert_segment_step(plant, track_start, track_end,
  airspeed_sp, altitude_sp, state, cfg, dt, wind)`. Wrapper `expert_goto_step`
  for single-waypoint use. Elevator sign-flip preserved (mirrors `cascade_pid.py:111`).

**Public API**:
```python
from jax_sim.controllers.expert import (
    expert_segment_step, expert_goto_step,
    ExpertConfig, ExpertState, ExpertDebug,
    default_expert_config, init_expert_state,
    save_expert_config, load_expert_config,
)
```

The primitive `expert_segment_step` matches PX4 NPFG's segment-based API
(start + end). The wrapper `expert_goto_step` synthesises a segment from
`expert_state.path_anchor` (set at `init_expert_state` time) to the target.

**Validation status** (with `tuned_expert_config.json` from default tuner):
- Waypoint litmus (0,0,-100) → (-100,-100,-100): closest XY 18.6 m at t≈27 s,
  altitude held within 5 m, airspeed within 2 m/s. PASS at the 30 m bar (the
  initial heading is opposite the target, requiring a full U-turn).
- Rate-step (cmd 0.5 rad/s): steady state 0.487, rise 40 ms, overshoot 3.5 %
  — matches the design `wn = 5 rad/s, zeta = 0.7`.
- Attitude-step (5°): measured τ 0.748 s vs design 0.8 s (6 % error).
- Single-step gradient: 32/32 ExpertConfig leaves finite.
- Multi-step BPTT (>5 steps) exhibits classical exploding gradients through
  the high-bandwidth closed loop — wrap with `jax.checkpoint` + per-step
  clipping if you need long-horizon backprop.

### Tuning (`jax_sim/controllers/tuning/`) — three approaches

**1. `es_tuner.py` — Evolution Strategy (vmap-parallel)**
- Population of `~2048` candidates, elite-selection mean update (`learning_rate=0.15`, `elite_ratio=0.1`).
- Two modes:
  - `tune_rate_only=True` (**default**): 9 params (rate Kp/Ki/Kd × 3 axes). Reference is a **single-frequency 0.5 Hz sinusoid** per axis at two throttles — easy for the optimizer to game (e.g. degenerate Kp≈0 / huge Ki solutions that phase-match the sine).
  - `tune_rate_only=False`: 14 params, including `tau_roll/pitch`, `speed_kp/ki`, `throttle_ff`. Reference is **step inputs** over 13 scenarios; cost = tracking + effort + crash + divergence penalties.
- All evaluation is `@jax.jit + jax.vmap`-able; the loss runs the *full* sim (not a surrogate).

**2. `model_tuner.py` — Model-based via Jacobian linearization**
- `jax.jacrev` over the closed-loop physics around a trim condition (per-axis) → first-order plant `ω̇ = a·ω + b·u`.
- Closed-form pole placement: `Kp = (2ζωₙ + a)/b`, `Ki = ωₙ²/b`. Handles sign of `b`, enforces a minimum-Kp floor.
- Output evaluated against the same ES loss for an apples-to-apples `final_loss`.
- This is the more principled tuner — it exploits the differentiable simulator that already exists.

**3. `expert_tuner.py` — Full-cascade jacrev pole placement**
- Reuses `model_tuner._rate_dynamics_from_linearization` for rate gains
  (`Kp = (2ζωₙ - a)/b`, `Ki = ωₙ²/b`, `FF = -a/b · ff_boost`).
- Attitude τ from inner-loop bandwidth via the `outer = ratio/inner` rule
  (default ratio=4 → τ=0.8s with ωₙ=5).
- TECS gains via pole placement on linearized longitudinal dynamics
  (∂(STE_rate)/∂(throttle), ∂(SEB_rate)/∂(pitch) around level cruise trim).
- NPFG defaults straight from PX4 yaml (period=1.0, damping=0.7).
- `tune_expert(...)` composes all four and writes `tuned_expert_config.json`.
- CLI: `uv run python scripts/tune_expert.py`. Flags:
  `--rate-wn`, `--rate-zeta`, `--rate-ff-boost`, `--attitude-ratio`,
  `--altitude-pole`, `--energy-pole`, `--airspeed-trim`.

The same `jacrev → analytical synthesis` recipe is the natural pattern
LQR/MPC implementations should follow. They get the linearization for free
via `_rate_dynamics_from_linearization` (and the longitudinal equivalent
inside `expert_tuner.design_tecs_gains`).

**Current `tuned_pid_config.json` is produced by `model_tuner` (`uv run python scripts/tune_pid.py --method model`).** Gains: `rate_kp ≈ [0.20, 0.20, 0.20]`, `rate_ki ≈ [8.08, 7.73, 7.36]`, `rate_kd = [0, 0, 0]`. All three Kp land on the `kp_target=0.2` floor — the bare-airframe rate pole is already at ≈ −18 rad/s, so the synthesizer raises `wn` to ≈ 37 rad/s to satisfy the Kp floor (closed-loop bandwidth ends up well above the user's `--rate-wn 8` request). Step responses track commands cleanly (30° roll → 30.0° steady-state). `tau_roll/pitch`, `speed_kp/ki`, `throttle_ff` are still at hand-picked defaults — `model_tuner` only handles the rate loop. A prior degenerate ES-rate-only config is preserved in `tuned_pid_config.json.bak` for reference.

### Guidance layer — does not exist

There is **no** position/heading/altitude guidance anywhere (no L1, no pure-pursuit, no TECS, no heading or altitude hold). The cascade PID's interface is `[roll, pitch, yaw_rate, speed]`. The only outer-outer loop in the repo is the RL policy in `FixedWingTarget-v1`, which is *supposed* to learn the `target_position → setpoints` mapping. **No trained checkpoint exists on disk** (`checkpoints/` is empty), so the system cannot currently fly to a waypoint autonomously without either (a) training PPO, or (b) adding a classical guidance controller.

### Environment (`jax_sim/env/`)

- `fixed_wing_target.py` defines `FixedWingTarget-v1`. `EnvState` is an immutable `NamedTuple` (`plane_state`, `target_pos`, `target_speed`, `pid_state`, `pid_config`, `time`, `last_action`, `wind_ned`, `turbulence_ned`, `key`). `reset` and `step` are `@jax.jit` and vmap-friendly.
- Constants: `DT = 0.004 s` (250 Hz), initial pos `[0, 0, -100]`, initial vel `[20, 0, 0]`, level quaternion. Action ∈ `[-1, 1]⁴` scaled to `±45° roll`, `±20° pitch`, `±15°/s yaw rate`, `±10 m/s` speed delta.
- **Observation: 19D**, normalized — relative target in body frame (/100m), body velocity (/30), speed error (/20), quaternion (4), body rates (/2), actuator states (4), last speed-delta action.
- **Reward**: alive bonus, distance shaping, speed-tracking, alignment, action-smoothness; **terminal**: −50 crash, +100 success.
- **Termination** (`termination.py`): crash (`z > 0`), success (`dist < 10 m` AND `|v_err| < 3 m/s`), OOB (`dist > 300 m`), timeout (`30 s` = 7500 steps).
- **Domain randomization** (`domain_randomization.py`): target distance 50–150 m + spherical sampling, target speed 15–25 m/s, PID gains ±30%. `randomize_physics_params` is **computed and discarded** (`del _phys_params` in `reset`) — wiring it through `AircraftParams` is a known gap.
- `wrappers.make_env` loads the tuned JSON, partials the wind config, and returns `{reset_fn, step_fn, obs_shape, action_shape, ...}`. **No vectorization here** — `vmap` is applied in `train.py`. `auto_reset_step` wraps `step` for auto-reset with terminal-obs bookkeeping.

### RL (`jax_sim/rl/`)

- **Flax NNX** (stateful), not `linen`. Actor and Critic are MLPs `(19) → [256, 256] tanh → head`. Actor output mean uses a small-init (0.01) final layer; `log_std` is a learnable **state-independent** parameter (init 0 → σ=1).
- `ppo.py`: standard PPO clip on the policy, value loss is MSE (the `clip_vloss` flag in `PPOConfig` is set but **not honored**), advantages normalized per minibatch, Gaussian entropy term (only adds if `ent_coef > 0`; default 0). GAE via `jax.lax.scan` in reverse.
- `train.py`: collect → GAE → update is a **Python loop** (not a single `jax.lax.scan`), explicitly for debuggability. `nnx.value_and_grad`; `optimizer.update` in-place. LR linearly annealed over `num_updates`.
- `buffers.py`: rollout buffer + `EpisodeStats`.
- `checkpoint.py`: pickled state dicts via `nnx.split` / `nnx.update`. Files at `checkpoints/<run>/checkpoint_<update>.pkl`. **Empty on disk currently.**
- `PPOConfig` is parsed by `tyro` from CLI. Defaults: 16 envs × 256 steps = 4096 batch, 8 minibatches, 10 epochs, `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.2`, 2M total timesteps.

### Other

- `logging/csv_logger.py` — trajectory and rate-debug CSVs; `viz/plots.py` — `sim_summary.png` and tuning history.
- `utils/quaternion.py` — the canonical quaternion ops (`rotate_vec_by_quat`, `quat_inv`, `quat_derivative`, `quat_to_euler_jax`). Use these rather than rolling new ones.
- `example_ppo_implementation.py` (repo root) is a **CleanRL Atari reference template** — discrete actions, Linen, CNN. Not imported anywhere; kept for comparison only.

## Conventions (AGENTS.md + observed)

- 4-space indent, `snake_case`. Public APIs are re-exported from each subpackage's `__init__.py`.
- Centralize physical constants in `physics/constants.py`; prefer extending `AircraftParams` to hardcoding.
- NED coordinates everywhere. GPU auto-picked by JAX.
- `runs/`, `pid_debug/`, `checkpoints/`, `*.png`, `*.csv` are generated artifacts — don't commit large outputs.
- When changing controller code or aero, re-run `tune_pid` (preferably full mode or `model_tuner`) and refresh `tuned_pid_config.json`. The current saved config is rate-only and degenerate.
- `physics_params` randomization is wired into `domain_randomization.py` but **not** plumbed into `equations_of_motion`. If you want sim-to-real-style training, wire it through `AircraftParams` first.

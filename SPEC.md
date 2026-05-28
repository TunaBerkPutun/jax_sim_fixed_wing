# AeroJax — Project Specification

> GPU-native, geometry-driven, differentiable 6-DOF flight dynamics for swarm-scale RL and gradient-based control on arbitrary aircraft configurations.

This document is the single source of truth for what AeroJax is, what it is not, and how the codebase will evolve. It is design-intent, not implementation. Code lives in the modules; this document explains *why* and *what shape*. Update this file when the design changes, not when the code changes.

---

## 1. Identity

**One-line pitch.** A geometry-in, dynamics-out flight simulator for fixed-wing, multirotor, tilt-rotor VTOL, and morphing-wing aircraft — running thousands of agents in parallel on a single GPU, with gradients flowing through the closed-loop rollout.

**The niche we own.** No existing tool combines:

| Property                       | JSBSim | PX4 SIH | Brax/MuJoCo | AeroSandbox | **AeroJax** |
| ------------------------------ | ------ | ------- | ----------- | ----------- | ----------- |
| Geometry-driven aero           |        | ✓       |             | ✓           | ✓           |
| Real-time forward simulation   | ✓      | ✓       | ✓           |             | ✓           |
| GPU / vectorized               |        |         | ✓           |             | ✓           |
| Multi-vehicle swarm scale      |        |         | ✓           |             | ✓           |
| RL training environment        | via wrapper | | ✓           |             | ✓           |
| Arbitrary configurations       |        | limited | N/A         | ✓           | ✓           |
| Differentiable closed-loop     |        |         | partial     | design-only | ✓           |

This is the genuine white space. The strategy is to claim it.

**Primary audiences.**
1. Multi-UAV RL research where aero matters (formation, pursuit-evasion, urban air mobility).
2. Control researchers who want gradient-based PID / MPC synthesis through real airfoil aero.
3. Exotic-airframe R&D (VTOL startups, morphing-wing teams) needing fast iteration without wind-tunnel data.

---

## 2. Goals and Non-Goals

### Goals

- **Geometry as the only required aircraft definition.** Wings, fuselage, propellers, control surfaces — described by their shape. No requirement on wind-tunnel tables or CFD data.
- **JAX/GPU-native throughout.** Every public function is `jit`-friendly and `vmap`-friendly. Throughput target: thousands of independent vehicles per GPU step.
- **Real-time forward simulation.** Step-by-step rollout via `jax.lax.scan` is the canonical evaluation mode.
- **Arbitrary aircraft configurations.** Fixed-wing, multirotor, tilt-rotor, tailsitter, canard, flying wing, morphing-wing — all expressed as collections of geometric primitives.
- **Differentiable closed-loop.** `jax.grad` through `obs → policy → controller → physics → reward` works without special-casing.
- **Honest validation story.** For every fidelity claim there is a reproducible cross-check against a higher-fidelity reference (JSBSim, XFOIL, OpenFOAM as appropriate).

### Non-goals

- **No piston, turbine, rocket, or buoyant-gas propulsion.** Electric only. The architecture stays open to adding others, but they are not on the roadmap.
- **No wind-tunnel-table aero pipeline.** We do not build a JSBSim XML loader. Tables conflict with the morphing/exotic-config goal and create a maintenance dead-end.
- **No conceptual-design / multidisciplinary-optimization layer.** AeroSandbox owns that. We consume design points, we do not optimize them at design time.
- **No WGS84 J2 gravity, Coriolis, ECI/ECEF frame stack.** Negligible effects at UAV scale and altitude.
- **No multi-step Adams-Bashforth integration.** RK4 + smooth dynamics already wastes less compute per unit accuracy and is more stable.
- **No CasADi or symbolic-graph optimization backend.** JAX tracing is the only differentiability mechanism.
- **No first-class flight visualization.** Plotting helpers exist; rendering pretty videos is not a project goal.

---

## 3. Scope

### In scope today

- 6-DOF rigid-body dynamics with quaternion attitude.
- Per-segment strip-theory aerodynamics with airfoil-aware sectional coefficients.
- Electric propulsion: direct-thrust prop (current) and Drela first-order BLDC with propeller (planned).
- Cascade PID autopilot with model-based gain synthesis via Jacobian linearization.
- PPO training loop with vmap-batched environments.
- Wind, gusts, MIL-spec turbulence.
- Smooth spring-damper ground contact.
- Forward-simulation, trim, and linearization tooling.

### Future scope (architecture must not preclude)

- Multi-agent / swarm environments with heterogeneous aircraft per batch.
- Communication / observation networks between agents.
- Hybrid dynamics (mode switches: vertical takeoff, transition, cruise).
- Differentiable MPC and model-based RL using the gradient-throughable rollout.
- Sensor models (IMU drift, GPS lag, camera/lidar — though rendering is out of scope).

### Hard out of scope

- Real-time visualization beyond matplotlib summaries.
- HIL (hardware-in-the-loop). The sim is software-only.
- Anything piston / combustion.
- High-supersonic (M > 0.9) aerodynamics.
- Re < 30k laminar-bubble regimes where XFOIL itself becomes unreliable.

---

## 4. Architectural Principles

1. **Everything is `jit`-traceable and `vmap`-friendly.** No `lax.cond` in the hot path — `jnp.where` and smooth blends only. No Python branches on tracers. No data-dependent shapes.
2. **Geometry is a `pytree`, not a class hierarchy.** AeroSandbox's `Wing(xsecs=[WingXSec(...)])` shape is the right *contract*, but it is implemented as flat dataclasses (`flax.struct.dataclass`) registered as pytrees, so a stack of heterogeneous aircraft can be `vmap`-ed cleanly.
3. **Aero, propulsion, and controllers are protocols, not subclasses.** Each is a callable with a known signature; concrete implementations swap in by composition. This is how a 1 kg UAV and a 5 kg quadplane share the same `dynamics.step` without code duplication.
4. **Differentiability is a property of every public function in physics/, controllers/, and env/.** When in doubt, prefer a smooth approximation over a piecewise hard switch. Document any gradient-blockers in the function docstring and the central hazards table.
5. **Validation is part of every module.** Each component has a `tests/` companion that compares its output against a reference (analytical limit, JSBSim, AeroSandbox AeroBuildup running on CPU, XFOIL polar).
6. **Default to honesty.** Where a model is a calibrated heuristic (Helmbold-Polhamus, Truong post-stall, etc.), say so in the code comments and the docs. Don't oversell.

---

## 5. System Architecture

```
                    ┌──────────────────────────────────────────┐
                    │   AircraftParams (flax.struct.dataclass) │
                    │  ┌────────────────────────────────────┐  │
                    │  │ MassProps, Geometry (Wing/Fuselage)│  │
                    │  │ AeroBackend (strip/buildup)        │  │
                    │  │ Propulsion (DirectThrust/BLDC+Prop)│  │
                    │  │ Atmosphere, ActuatorParams         │  │
                    │  └────────────────────────────────────┘  │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
┌──────────────┐    ┌──────────────────────────────────────┐    ┌──────────────┐
│ obs (19D)    │───▶│ Policy (Flax NNX MLP, learnable)     │───▶│ action (4D)  │
└──────────────┘    └──────────────────────────────────────┘    └──────┬───────┘
                                                                       │
                                         ┌─────────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  Cascade PID                             │
                    │  (attitude → rate → speed; model-tuned)  │
                    └────────────────────┬─────────────────────┘
                                         │ surface_cmds, throttle
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  Actuator dynamics (lag filters)         │
                    │  ↓                                       │
                    │  Aero: NeuralFoil(airfoil) →             │
                    │        Helmbold-Polhamus 3D correction → │
                    │        strip integration over wings,     │
                    │        Jorgensen fuselage,               │
                    │        BEMT/disk-theory rotors,          │
                    │        propeller slipstream on tail/fin  │
                    │  Propulsion: BLDC torque, prop thrust    │
                    │  Atmosphere + wind/turbulence            │
                    │  ↓                                       │
                    │  Forces / moments → RK4 rigid-body       │
                    │  Ground: smooth spring-damper            │
                    └────────────────────┬─────────────────────┘
                                         │ next_state
                                         ▼
                                     reward + obs
```

Every arrow above is JAX-native. The whole pipeline is a single traced function under `jit`, `vmap`-able across a batch of (`AircraftParams`, initial state, policy params) tuples.

---

## 6. Module Specifications

### 6.1 Geometry (`aerojax/geometry/`)

Adopt the AeroSandbox object shape, implement as flat pytrees.

- **`Airfoil`**: identified by its 16-coefficient Kulfan/CST shape parameterization (upper/lower) plus LE radius weight and TE thickness. UIUC and NACA airfoils are convenience constructors that emit Kulfan coefficients. Stored as a leaf in the aircraft pytree so different segments can use different airfoils.
- **`WingXSec`**: `(xyz_le, chord, twist, airfoil, control_surfaces)`. Defines one station of a lofted wing.
- **`Wing`**: ordered list of `WingXSec` + a `symmetric` flag (mirror about XZ for symmetric aircraft). A wing is a collection of strips between adjacent xsecs.
- **`ControlSurface`**: `(hinge_chord_fraction, deflection_index_into_controls, symmetric, trailing_edge)`. Multiple per `WingXSec` allowed (aileron + flap + spoiler all on the same wing strip).
- **`Fuselage`**: ordered list of `FuselageXSec` (centerline position, cross-section area, equivalent diameter). Used by Jorgensen viscous-crossflow.
- **`Propulsor`**: `(position, orientation, propeller_geometry, motor)`. Tilt-rotor support: orientation is a function of a tilt actuator state.
- **`Aircraft`**: top-level pytree containing `wings: list[Wing]`, `fuselages: list[Fuselage]`, `propulsors: list[Propulsor]`, plus reference quantities (`s_ref`, `b_ref`, `c_ref`, `xyz_ref`).

The flat list shape is critical: a multirotor is `Aircraft(wings=[], fuselages=[fuselage_pod], propulsors=[p1,p2,p3,p4])`. A tilt-rotor is the same with `propulsors[i].orientation` driven by a tilt state. A morphing-wing aircraft has `Wing.xsecs[i].chord` or `.twist` driven by actuator state. No special cases.

### 6.2 Airfoil Model: NeuralFoil-in-JAX (`aerojax/airfoil/`)

The gating capability. A small physics-informed MLP that maps `(Kulfan_coefficients, alpha, Re, n_crit, xtr) → (Cl, Cd, Cm, BL_params, analysis_confidence)` for any subsonic airfoil.

- **Source**: NeuralFoil (Sharpe, MIT, MIT-licensed, [github.com/peterdsharpe/NeuralFoil](https://github.com/peterdsharpe/NeuralFoil)).
- **Architecture (read the source, not the summary)**: it is not "just an MLP." The forward path is `kulfan → latent encoder → MLP (dense + tanh layers) → output decoder → empirical-fusion → final coefficients`, plus a parallel Mahalanobis-distance computation against the fitted training-input Gaussian that produces an `analysis_confidence` output. Weights are pre-trained `.npz` files (8 model sizes, MIT-licensed, ship in the repo). PyTorch is only used for training; **runtime is pure NumPy**, which maps to JAX trivially.
- **Adopt vs. retrain**:
  - **Path A — adopt weights**: re-implement the wrapper in JAX, load the `.npz` weights as JAX arrays, cite NeuralFoil + Sharpe's PhD thesis. Fast, license-clean (MIT). Default path.
  - **Path B — retrain**: training data is not yet public (Sharpe is setting up Git LFS); contact him for access, or sweep XFOIL ourselves over UIUC + Kulfan perturbations. Only worth it before public release for clean dataset provenance.
- **Default model size**: `"large"` (4 layers × 128 wide). Author's explicit advice in README: do not push past `large` — bigger models start overfitting XFOIL's `C¹`-discontinuities at transition, which hurts gradient-based optimization. Smaller models (`medium`, `small`) are options for swarm-throughput scaling.
- **API**: `airfoil_aero(kulfan, alpha, Re) -> dict[Cl, Cd, Cm, analysis_confidence, ...]`. Pure JAX, jit/vmap clean.
- **Post-stall**: Truong (2020) analytical blend handles 360-degree alpha — pre-stall NeuralFoil output, post-stall flat-plate-like normal/tangential decomposition, smooth `tanh` join. Replaces our current generic `tanh` stall blend.
- **Compressibility**: Laitone subsonic correction (higher-order Prandtl-Glauert / Karman-Tsien) applied to NeuralFoil output for Mach > 0.3. Drag-divergence Mach captured implicitly via `C_p,min`.
- **No AeroSandbox runtime dependency.** NeuralFoil's `main.py` imports `aerosandbox` at the top, but the inference path doesn't need it. We vendor our own Kulfan/CST parameterization (~50 lines of documented formulas).
- **Bonus: Linear CL model.** NeuralFoil ships a second mode where CL is affine in alpha (still nonlinear in shape/Re). Faster than the full NN; the affine structure lets the trim solver and lifting-line solvers run as one-shot linear solves. Useful for extreme-swarm scenarios.

### 6.3 Aerodynamics: Strip + Buildup (`aerojax/aero/`)

Per-segment strip theory with airfoil-aware coefficients and validated 3D corrections.

- **Per strip**: compute local velocity (freestream + body rotation × position + propeller slipstream if applicable). Compute local `alpha`, `beta`, Re, Mach. Apply Helmbold-Polhamus finite-wing correction to map 2D `Cl/Cd/Cm` from NeuralFoil into 3D `CL/CD/CM` contribution.
- **Finite-wing correction**: closed-form `CL/Cl = AR / (2 + sqrt(4 + (AR β / η)² + (AR tan(sweep) / η)²))`. Smooth, differentiable, validated. Source: Raymer §12.4.1 citing DATCOM.
- **Induced drag**: Trefftz-plane `CDi = CL² / (π · AR_effective · e)` with Nita-Scholz Oswald efficiency `e(taper, AR, sweep)`. Smooth and AR/morphing-aware.
- **Fuselage**: Jorgensen (NASA TR R-474) slender-body normal force + viscous crossflow + Hoerner skin-friction form factor. Closed-form, fast, gives non-zero CY/Cn at sideslip.
- **Control surfaces**: enter the segment's local alpha via a hinge-effectiveness function (XFoil-fit). Multiple surfaces per strip sum linearly in their delta-alpha contributions.
- **Output**: total `F_body`, `M_body` from the aerodynamic system. Drop-in replacement for current `compute_fixed_wing_aero`.

### 6.4 Rotor Model (`aerojax/rotor/`)

Geometry-driven rotor aero, the same way wings are geometry-driven.

- **Inputs**: blade chord and twist distributions (functions of radial fraction), root cutout, number of blades, rotor RPM, freestream velocity in rotor frame.
- **Method**: blade-element momentum theory (BEMT). For each radial station, compute local Re, local angle, query airfoil model (same NeuralFoil as wings) for `Cl, Cd`. Solve for induced inflow via momentum balance. Integrate to thrust and torque.
- **Slipstream**: same `sqrt(2T/(ρπr²))` actuator-disk velocity boost on downstream segments as the current code, but referenced from the rotor's geometry instead of being hand-keyed to the tailplane.
- **Why this matters**: unifies fixed-wing propellers and multirotor rotors under one abstraction. Same physics; the difference is whether the rotor is fixed (multirotor) or has surfaces downstream (tractor prop).

### 6.5 Propulsion: BLDC (`aerojax/propulsion/`)

Replace the linear `T = throttle · T_max` with a real electric drivetrain when fidelity matters.

- **Motor model**: Drela first-order BLDC. Inputs: throttle, battery voltage, RPM. Outputs: shaft torque, current draw. Parameters: kV, R_internal, I0. Differentiable algebraic solve (no iteration in the hot path).
- **Battery**: optional. Constant voltage to start; add depletion/sag in a later phase.
- **Coupling to rotor**: rotor consumes shaft torque and returns thrust + reaction torque + drag torque, driving the algebraic equilibrium `shaft_torque = rotor_torque(RPM, V, atm)` to compute RPM.
- **Backward compatibility**: `DirectThrustProp` (current model) stays as the simplest implementation of the `Propulsion` protocol. New aircraft opt into `BLDCWithRotor` by setting it in `AircraftParams`.

### 6.6 Atmosphere (`aerojax/atmosphere/`)

Differentiable ISA across the full troposphere/stratosphere.

- **Form**: AeroSandbox's log-pressure-spline approach. C¹-continuous fit to 1976 COESA standard atmosphere, accuracy <0.1% over 0–30 km, smooth across the tropopause.
- **Outputs**: density, pressure, temperature, speed of sound, dynamic viscosity. All differentiable, vectorized.

### 6.7 Rigid-Body Dynamics (`aerojax/physics/`)

Mostly inherited from the current sim, with a small cleanup pass.

- **State**: `(pos[3], vel[3], quat[4], omega[3], actuator_state[N])`. NED frame. Quaternion is body→earth.
- **Integrator**: RK4 on `[pos, vel, quat, omega]`. Forces evaluated once per outer step and held across substeps (cheap, standard).
- **Quaternion**: post-step renormalization; clamp norm at `1e-8`.
- **Gravity**: constant `9.81` in earth-frame `+z` (NED down).
- **Body rates**: clipped at `±max_body_rate`.
- **Gyroscopic angular momentum**: extend `MassProps` to carry stored angular momentum `h_body` from spinning propellers; include `omega × h_body` term in the rotational dynamics. Currently absent; matters for VTOL.
- **No Earth-rotation or J2 terms.** Documented as out of scope.

### 6.8 Ground Contact (`aerojax/physics/ground.py`)

Replace the current `if z >= 0: zero everything` with a smooth, differentiable spring-damper.

- **Per contact point** (defined in `AircraftParams.contact_points`): spring force on penetration depth, damper force on penetration rate.
- **Friction**: separate static / dynamic friction (smooth blend between the two via a tanh of slip velocity).
- **Differentiable**: C¹ smooth at zero penetration via softplus, so gradients flow through touchdown.
- **Why**: enables RL on landing tasks and removes the gradient-killing `lax.select` flagged in current `CLAUDE.md`.

### 6.9 Wind and Turbulence (`aerojax/wind/`)

Upgrade the existing per-axis OU-Dryden to MIL-F-8785C.

- **Steady wind**: NED bias vector. Unchanged.
- **Discrete gust**: 1-minus-cosine over configurable duration. Unchanged.
- **Continuous turbulence**: Dryden spectrum with altitude-scaled length scales (below 1000 ft AGL: `L_w = h`, `L_u = h / (0.177 + 0.000823·h)^1.2`; above 2000 ft: 1750 m). Per MIL-F-8785C.
- **Sigma scaling**: from wind speed at 20 ft AGL via probability-of-exceedance.
- **Angular turbulence**: include `p`, `q`, `r` disturbances driven by `sigma_w` and effective span/MAC. Currently absent; visible in inner rate-loop response.

### 6.10 Controllers (`aerojax/controllers/`)

Existing cascade PID stays. Tuning gets a JAX-native trim solver.

- **Cascade PID**: attitude (outer, pure-P) → rate (inner, full PID with anti-windup) → speed (PI + feedforward). Unchanged in structure.
- **Trim solver (new)**: `find_trim(aircraft, condition) -> (alpha, controls)` solving steady-state residuals (force, moment balance). Implementation: bisection per axis (like JSBSim) wrapped in `lax.while_loop` so it is `vmap`-able and runs under `jit`. Replaces the current scipy `fsolve` call in `extract_aero_derivs.py`.
- **Linearization**: keep using `jax.jacrev`. The model_tuner approach already works; no change beyond pointing it at the new aero stack.

### 6.11 Environment (`aerojax/env/`)

Existing `FixedWingTarget-v1` stays. Architectural changes:

- **Wire `physics_params` randomization through `AircraftParams`.** Currently computed and discarded. Fix is one-line plumbing.
- **Heterogeneous-batch support.** A vectorized env can carry a stack of `AircraftParams` where each agent has different geometry. Foundation for swarm RL with mixed aircraft.
- **Multi-agent extension (future).** Single-agent env stays the canonical baseline. Multi-agent wrapper composes N single-agent envs with shared observation channels.

### 6.12 RL Training (`aerojax/rl/`)

PPO loop is fine. No structural changes planned.

- **Continue with Flax NNX**, not Linen. Continue with stateful actor/critic.
- **Throughput targets**: characterize at batch sizes 1, 64, 1024, 4096 across GPU tiers. Publish numbers in the README.
- **Save trained checkpoints in `checkpoints/`** with reproducible seeds; currently empty.

---

## 7. Validation Strategy

Each fidelity claim is backed by a specific cross-check. Validation is *part of the deliverable* for each phase, not a separate phase.

| Claim                                              | Reference                                              | Cadence                          |
| -------------------------------------------------- | ------------------------------------------------------ | -------------------------------- |
| Airfoil Cl/Cd/Cm match XFOIL                       | XFOIL polars for ~10 reference airfoils                | Per NeuralFoil training run      |
| 3D wing CL matches lifting-line + AeroBuildup      | AeroSandbox `AeroBuildup` on the same geometry (CPU)   | On every aero refactor           |
| 6-DOF trajectory matches reference for matched a/c | JSBSim with our matched XML (already built)            | On every dynamics refactor       |
| Rigid-body integration is correct                  | Closed-form ballistic + Euler-rotation tests           | One-time, in tests/              |
| Ground contact is smooth and stable                | Compare against analytical drop test                   | One-time, in tests/              |
| Turbulence matches MIL spec                        | PSD of generated time series vs Dryden analytical PSD  | One-time, in tests/              |
| Trim solver converges and matches reference        | Compare against scipy fsolve for representative cases  | One-time, in tests/              |
| Throughput meets target                            | `benchmark_env.py` at multiple batch sizes             | Per release                      |

The matched-JSBSim experiment we already ran is the **anchor regression test** — any change that breaks the trim-overlay agreement is a regression.

---

## 8. Risks and Gating Items

| Risk                                                              | Mitigation                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------------- |
| NeuralFoil port misses the wrapper (latent encoder, fusion, etc.) | Read the whitepaper + source before porting. The MLP layers alone are not the full model. |
| Training data not yet public (Path B blocked)                     | Path A (adopt weights, MIT-licensed) is unblocked. Path B is a Phase-9-or-later concern. |
| Throughput target not met at swarm scale                          | Profile early; the gating items are usually Python loops      |
| 3D approximation accuracy worse than expected on unusual configs  | Validation against AeroSandbox AeroBuildup on the same geom   |
| Differentiability blocked by ground contact / lookup discontinuity| Smooth replacements are documented per module                 |
| Project bloat: scope creep into design optimization               | Re-read this spec when tempted; design optimization is non-goal |

**NeuralFoil-in-JAX is the single biggest gating item.** It is in Phase 2, immediately after the foundation refactor. If it does not port cleanly, the whole strategy must be reconsidered.

---

## 9. Phased Roadmap

Each phase produces a runnable artifact and a validation result. No phase is "code only." Detailed task breakdown lives in the project task tracker.

- **Phase 0 — Repo hygiene and identity.** Rename, README, license, packaging, public-facing first impression.
- **Phase 1 — Foundation refactor.** Geometry pytree hierarchy, protocol-based aero/propulsion, `lax.cond → jnp.where` cleanup, `physics_params` randomization wired through.
- **Phase 2 — NeuralFoil-in-JAX.** The gating capability. Port architecture + weights; validate against XFOIL polars.
- **Phase 3 — AeroBuildup port.** Strip theory + Helmbold-Polhamus 3D correction + Nita-Scholz Oswald + Jorgensen fuselage + Truong post-stall. Validate against AeroSandbox CPU reference.
- **Phase 4 — Rotor + BLDC + atmosphere.** BEMT rotor, Drela BLDC motor, log-pressure-spline atmosphere. Unifies multirotor + fixed-wing prop under one abstraction.
- **Phase 5 — Ground contact + wind upgrade.** Smooth spring-damper landing gear, MIL-F-8785C turbulence with angular components.
- **Phase 6 — Trim + linearization tooling.** JAX-native bisection trim solver, vmap-able. Hooks into model_tuner.
- **Phase 7 — Swarm scaling and benchmarks.** Heterogeneous-batch envs, throughput characterization, multi-agent wrapper.
- **Phase 8 — RL baselines and the differentiable-control paper.** Train PPO at swarm scale. Write up Jacobian-linearization PID synthesis through real airfoil aero.
- **Phase 9 — Public release.** Documentation, examples, PyPI package, paper preprint.

Phases 0–6 are foundation. Phases 7–9 are product and publication. Re-evaluate the roadmap at the end of Phase 3 (post-NeuralFoil) and again at Phase 6 (post-foundation).

"""Evolution Strategy (ES) optimizer for PID tuning.

Uses simple ES with elite selection to find optimal PID gains.
Leverages JAX's vmap for parallel population evaluation.
"""

import json
from typing import NamedTuple, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jax_sim.controllers.tuning.loss import (
    evaluate_population,
    evaluate_rate_population,
    params_to_config,
    rate_params_to_config,
    get_initial_params,
    get_rate_initial_params,
    get_param_bounds,
    get_rate_param_bounds,
    get_rate_scenario_keys,
    sample_rate_amplitude,
    simulate_rate_scenario_debug,
    RATE_AXES,
    RATE_THROTTLE_CMDS,
    N_PARAMS,
    N_RATE_PARAMS,
)
from jax_sim.controllers.pid_gains import PIDConfig
from jax_sim.logging.csv_logger import save_rate_debug_csv


class TuningResult(NamedTuple):
    """Result of ES tuning."""

    params: jnp.ndarray
    config: PIDConfig
    final_loss: float
    loss_history: jnp.ndarray


def run_es_tuning(
    pop_size: int = 2048,
    generations: int = 50,
    sigma: float = 0.1,
    learning_rate: float = 0.15,
    elite_ratio: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
    tune_rate_only: bool = True,
    debug_nonfinite: bool = False,
    debug_output_dir: str = "pid_debug",
    debug_log_limit: int = 5,
) -> TuningResult:
    """Find optimal PID gains using Evolution Strategy.

    Algorithm:
    1. Start with initial parameter guess (mean)
    2. For each generation:
       - Generate population by adding Gaussian noise to mean
       - Evaluate all candidates in parallel (vmap)
       - Select top elite_ratio% as "elites"
       - Move mean toward elite average
    3. Return best parameters found

    Args:
        pop_size: Population size (number of candidates per generation)
        generations: Number of generations to evolve
        sigma: Mutation standard deviation (exploration noise)
        learning_rate: How fast to move mean toward elites
        elite_ratio: Fraction of population to use as elites (0.1 = top 10%)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        TuningResult with optimal params, config, and loss history
    """
    key = random.PRNGKey(seed)
    if tune_rate_only:
        bounds = get_rate_param_bounds()
        mean_params = get_rate_initial_params()
        n_params = N_RATE_PARAMS
        evaluate_fn = evaluate_rate_population
    else:
        bounds = get_param_bounds()
        mean_params = get_initial_params()
        n_params = N_PARAMS
        evaluate_fn = evaluate_population

    # Compute param-specific sigma based on bounds range
    param_range = bounds[:, 1] - bounds[:, 0]
    sigma_scaled = sigma * param_range

    elite_count = max(1, int(pop_size * elite_ratio))
    loss_history = []
    best_loss = float("inf")
    best_params = mean_params

    if verbose:
        print(f"ES Auto-Tuning: {pop_size} candidates × {generations} generations")
        print(f"Elite count: {elite_count}, σ = {sigma}, lr = {learning_rate}")
        print("-" * 60)

    debug_logs_written = 0

    for gen in range(generations):
        key, subkey = random.split(key)

        # 1. Generate population (mutate around mean)
        noise = random.normal(subkey, (pop_size, n_params))
        population = mean_params + sigma_scaled * noise

        # Clip to bounds
        population = jnp.clip(population, bounds[:, 0], bounds[:, 1])

        # 2. Evaluate all candidates in parallel
        losses = evaluate_fn(population, subkey)
        losses_np = np.array(jax.device_get(losses))

        # 3. Select elites (lowest loss)
        sorted_indices = jnp.argsort(losses)
        elite_indices = sorted_indices[:elite_count]
        elites = population[elite_indices]
        elite_losses = losses[elite_indices]

        gen_best_loss = elite_losses[0]
        gen_best_params = elites[0]
        gen_mean_loss = jnp.mean(losses)

        # Track global best
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_params = gen_best_params

        loss_history.append(float(gen_best_loss))

        # 4. Update mean toward elites
        elite_mean = jnp.mean(elites, axis=0)
        mean_params = mean_params + learning_rate * (elite_mean - mean_params)

        # Ensure mean stays in bounds
        mean_params = jnp.clip(mean_params, bounds[:, 0], bounds[:, 1])

        if debug_nonfinite and debug_logs_written < debug_log_limit:
            nonfinite_idx = np.where(~np.isfinite(losses_np))[0]
            if nonfinite_idx.size > 0 and tune_rate_only:
                idx = int(nonfinite_idx[0])
                params = population[idx]
                config = rate_params_to_config(params)
                scenario_keys = get_rate_scenario_keys(subkey)
                key_idx = 0

                output_dir = Path(debug_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                for throttle_cmd in RATE_THROTTLE_CMDS:
                    for axis in RATE_AXES:
                        amplitude = sample_rate_amplitude(scenario_keys[key_idx])
                        debug = simulate_rate_scenario_debug(
                            config,
                            axis=axis,
                            throttle_cmd=float(throttle_cmd),
                            amplitude=float(amplitude),
                        )
                        total_loss = float(debug["total_loss"])
                        if not np.isfinite(total_loss):
                            time = np.arange(debug["states"].shape[0]) * 0.004
                            rates = debug["states"][:, 10:13]
                            actuators = debug["actuators"]
                            filename = (
                                f"rate_debug_gen{gen}_idx{idx}_axis{axis}_thr{throttle_cmd:.1f}.csv"
                            )
                            save_rate_debug_csv(
                                time,
                                debug["rate_sp"],
                                rates,
                                actuators,
                                filepath=str(output_dir / filename),
                            )
                            debug_logs_written += 1
                            break
                        key_idx += 1
                    if debug_logs_written >= debug_log_limit:
                        break

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            if tune_rate_only:
                print(f"Gen {gen:3d}: Best Loss = {gen_best_loss:8.2f} | "
                      f"Mean Loss = {float(gen_mean_loss):8.2f} | "
                      f"Kp=[{gen_best_params[0]:.2f},{gen_best_params[1]:.2f},{gen_best_params[2]:.2f}] "
                      f"Ki=[{gen_best_params[3]:.2f},{gen_best_params[4]:.2f},{gen_best_params[5]:.2f}] "
                      f"Kd=[{gen_best_params[6]:.3f},{gen_best_params[7]:.3f},{gen_best_params[8]:.3f}]")
            else:
                print(f"Gen {gen:3d}: Best Loss = {gen_best_loss:8.2f} | "
                      f"Mean Loss = {float(gen_mean_loss):8.2f} | "
                      f"τ_r={gen_best_params[0]:.3f} τ_p={gen_best_params[1]:.3f} "
                      f"Kp=[{gen_best_params[2]:.2f},{gen_best_params[3]:.2f},{gen_best_params[4]:.2f}]")

    # Convert best params to config
    best_config = rate_params_to_config(best_params) if tune_rate_only else params_to_config(best_params)

    if verbose:
        print("-" * 60)
        print("TUNED PID PARAMETERS:")
        print(f"  tau_roll:    {best_config.tau_roll:.3f}s")
        print(f"  tau_pitch:   {best_config.tau_pitch:.3f}s")
        print(f"  rate_kp:     [{best_config.rate_kp[0]:.3f}, "
              f"{best_config.rate_kp[1]:.3f}, {best_config.rate_kp[2]:.3f}]")
        print(f"  rate_ki:     [{best_config.rate_ki[0]:.3f}, "
              f"{best_config.rate_ki[1]:.3f}, {best_config.rate_ki[2]:.3f}]")
        print(f"  rate_kd:     [{best_config.rate_kd[0]:.4f}, "
              f"{best_config.rate_kd[1]:.4f}, {best_config.rate_kd[2]:.4f}]")
        print(f"  speed_kp:    {best_config.speed_kp:.3f}")
        print(f"  speed_ki:    {best_config.speed_ki:.3f}")
        print(f"  throttle_ff: {best_config.throttle_ff:.3f}")
        print(f"\nFinal Loss: {best_loss:.2f}")

    return TuningResult(
        params=best_params,
        config=best_config,
        final_loss=float(best_loss),
        loss_history=jnp.array(loss_history),
    )


def save_tuned_config(result: TuningResult, filepath: str = "tuned_pid_config.json"):
    """Save tuned PID config to JSON file.

    Args:
        result: TuningResult from run_es_tuning
        filepath: Output file path
    """
    config = result.config
    data = {
        "tau_roll": float(config.tau_roll),
        "tau_pitch": float(config.tau_pitch),
        "rate_kp": [float(x) for x in config.rate_kp],
        "rate_ki": [float(x) for x in config.rate_ki],
        "rate_kd": [float(x) for x in config.rate_kd],
        "speed_kp": float(config.speed_kp),
        "speed_ki": float(config.speed_ki),
        "throttle_ff": float(config.throttle_ff),
        "rate_limit": float(config.rate_limit),
        "integral_limit": float(config.integral_limit),
        "final_loss": result.final_loss,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to: {filepath}")


def load_tuned_config(filepath: str = "tuned_pid_config.json") -> PIDConfig:
    """Load tuned PID config from JSON file.

    Args:
        filepath: Input file path

    Returns:
        PIDConfig with loaded parameters
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return PIDConfig(
        tau_roll=data["tau_roll"],
        tau_pitch=data["tau_pitch"],
        rate_kp=jnp.array(data["rate_kp"]),
        rate_ki=jnp.array(data["rate_ki"]),
        rate_kd=jnp.array(data["rate_kd"]),
        speed_kp=data["speed_kp"],
        speed_ki=data["speed_ki"],
        throttle_ff=data["throttle_ff"],
        rate_limit=data.get("rate_limit", 1.047),
        integral_limit=data.get("integral_limit", 2.0),
    )

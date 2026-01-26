# Repository Guidelines

## Project Structure & Module Organization
- `jax_sim/` is the core package.
  - `controllers/`, `env/`, `physics/`, `rl/`, `logging/`, `utils/`, `viz/` hold the main simulation, control, and RL modules.
- `scripts/` contains runnable entry points for simulations, training, tuning, and tests.
- `runs/`, `pid_debug/`, `*.png`, `*.csv` are generated artifacts from training, tuning, and evaluations. Avoid committing new large outputs unless explicitly needed.
- Top-level configs like `tuned_pid_config.json` capture tuned controller parameters.

## Build, Test, and Development Commands
- `uv run python scripts/run_sim.py` runs a default fixed-wing simulation and writes `sim_log.csv` and `sim_summary.png`.
- `uv run python scripts/train_ppo.py --num-envs 32 --learning-rate 1e-4` starts PPO training; outputs to `runs/` and `checkpoints/`.
- `uv run python scripts/tune_pid.py --pop-size 4096 --generations 100` tunes PID gains and writes a config JSON plus `tuning_history.png`.
- `uv run python scripts/test_env.py` runs environment sanity checks and a random-policy episode.
- `uv run python scripts/test_pid.py --roll-cmd 10 --duration 5` simulates PID tracking and writes plots/CSV logs.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation, module names in `snake_case`, and public APIs grouped in `__init__.py`.
- Use concise docstrings for public functions and keep numerical constants centralized (e.g., `jax_sim/physics/constants.py`).
- No formatter/linter is configured; follow existing style and keep lines readable.

## Testing Guidelines
- No pytest/coverage configuration exists. Use the script-based checks in `scripts/` for validation.
- Prefer adding new test scripts alongside related functionality (e.g., `scripts/test_env.py` for environment behaviors).

## Commit & Pull Request Guidelines
- This repository has no commit history yet, so there is no established message convention. Use clear, imperative summaries (e.g., "Add PPO rollout buffer") and include scope if helpful.
- PRs should describe the change, provide reproduction commands, and attach plots or logs when modifying controllers or training pipelines.

## Configuration & Runtime Notes
- Simulation defaults assume NED coordinates and JAX execution; expect GPU usage when available.
- Keep tuned configs (`tuned_pid_config.json`) and plots up to date when changing controller logic.

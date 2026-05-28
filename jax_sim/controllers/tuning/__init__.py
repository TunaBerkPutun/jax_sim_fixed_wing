"""PID auto-tuning using Evolution Strategy."""

from jax_sim.controllers.tuning.loss import (
    evaluate_pid_config,
    params_to_config,
    config_to_params,
    config_to_rate_params,
    get_param_bounds,
)
from jax_sim.controllers.tuning.es_tuner import (
    run_es_tuning,
    TuningResult,
)
from jax_sim.controllers.tuning.model_tuner import run_model_tuning_rate
from jax_sim.controllers.tuning.expert_tuner import (
    design_attitude_tau,
    design_npfg_defaults,
    design_rate_gains,
    design_tecs_gains,
    tune_expert,
)

__all__ = [
    "evaluate_pid_config",
    "params_to_config",
    "config_to_params",
    "config_to_rate_params",
    "get_param_bounds",
    "run_es_tuning",
    "run_model_tuning_rate",
    "TuningResult",
    # Expert cascade tuner
    "design_rate_gains",
    "design_attitude_tau",
    "design_tecs_gains",
    "design_npfg_defaults",
    "tune_expert",
]

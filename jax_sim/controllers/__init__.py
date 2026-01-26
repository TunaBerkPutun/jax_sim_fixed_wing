"""Control algorithms for UAV.

Main entry point: cascade_pid_step
"""

from jax_sim.controllers.pid_gains import (
    PIDConfig,
    PIDState,
    create_pid_config,
    create_pid_state,
    randomize_pid_gains,
)
from jax_sim.controllers.cascade_pid import (
    cascade_pid_step,
    init_pid_state,
    reset_pid_state,
)
from jax_sim.controllers.attitude.pid import (
    attitude_controller,
    attitude_controller_vectorized,
)
from jax_sim.controllers.rate.pid import (
    rate_controller,
    rate_controller_single_axis,
)
from jax_sim.controllers.speed.pid import (
    speed_controller,
    speed_controller_simple,
)

__all__ = [
    # Main entry point
    "cascade_pid_step",
    "init_pid_state",
    "reset_pid_state",
    # Configuration
    "PIDConfig",
    "PIDState",
    "create_pid_config",
    "create_pid_state",
    "randomize_pid_gains",
    # Individual controllers
    "attitude_controller",
    "attitude_controller_vectorized",
    "rate_controller",
    "rate_controller_single_axis",
    "speed_controller",
    "speed_controller_simple",
]

#!/usr/bin/env python3
"""Run fixed-wing UAV simulation."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jax_sim.physics.dynamics import equations_of_motion
from jax_sim.logging.csv_logger import save_trajectory_csv
from jax_sim.viz.plots import load_log, plot_summary


def run_simulation(
    duration: float = 10.0,
    dt: float = 0.004,
    controls: tuple = (0.0, -0.5, 0.0, 1.0),
    output_csv: str = "sim_log.csv",
    output_plot: str = "sim_summary.png",
):
    """Run the UAV simulation.

    Args:
        duration: Simulation duration (seconds)
        dt: Timestep (seconds)
        controls: Control inputs [aileron, elevator, rudder, throttle]
        output_csv: Output CSV file path
        output_plot: Output plot file path

    Returns:
        final_state: Final state vector
        trajectory: Full state history
    """
    # Initial state
    # Pos: [0,0,-100] (100m altitude), Vel: [20,0,0] (20m/s forward)
    # Quat: [1,0,0,0] (level), Omega: [0,0,0]
    state0 = jnp.array(
        [
            0.0, 0.0, -100.0,  # Position (NED)
            20.0, 0.0, 0.0,     # Velocity (NED)
            1.0, 0.0, 0.0, 0.0, # Quaternion
            0.0, 0.0, 0.0,      # Angular velocity
            0.0, 0.0, 0.0, 0.0, # Actuator states
        ]
    )

    controls_arr = jnp.array(controls)
    steps = int(duration / dt)

    # Scan function (JAX loop)
    def step_fn(carry, _):
        s = carry
        next_s = equations_of_motion(s, controls_arr, dt)
        return next_s, next_s

    final_state, trajectory = jax.lax.scan(step_fn, state0, jnp.arange(steps))

    # Save CSV log
    save_trajectory_csv(trajectory, dt, log_dt=0.1, filepath=output_csv)
    print(f"Saved log to {output_csv}")

    # Generate summary plot
    data = load_log(output_csv)
    plot_summary(data, out_path=output_plot)
    print(f"Saved plot to {output_plot}")

    print(f"Simulation complete. Final position: {final_state[0:3]}")

    return final_state, trajectory


def main():
    parser = argparse.ArgumentParser(description="Run UAV simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--dt", type=float, default=0.004, help="Timestep (s)")
    parser.add_argument("--aileron", type=float, default=0.0, help="Aileron command (-1 to 1)")
    parser.add_argument("--elevator", type=float, default=-0.5, help="Elevator command (-1 to 1)")
    parser.add_argument("--rudder", type=float, default=0.0, help="Rudder command (-1 to 1)")
    parser.add_argument("--throttle", type=float, default=1.0, help="Throttle command (0 to 1)")
    parser.add_argument("--csv", default="sim_log.csv", help="Output CSV path")
    parser.add_argument("--plot", default="sim_summary.png", help="Output plot path")
    args = parser.parse_args()

    run_simulation(
        duration=args.duration,
        dt=args.dt,
        controls=(args.aileron, args.elevator, args.rudder, args.throttle),
        output_csv=args.csv,
        output_plot=args.plot,
    )


if __name__ == "__main__":
    main()

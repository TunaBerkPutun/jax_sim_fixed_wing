#!/usr/bin/env python3
"""Test script for the Cascade PID controller.

Runs a simulation with the PID controller and plots setpoint tracking.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_sim.physics.dynamics import equations_of_motion
from jax_sim.controllers import (
    cascade_pid_step,
    create_pid_config,
    create_pid_state,
    PIDConfig,
    PIDState,
)
from jax_sim.controllers.attitude.pid import attitude_controller
from jax_sim.controllers.tuning.es_tuner import load_tuned_config
from jax_sim.utils.quaternion import quat_to_euler_jax
from jax_sim.logging.csv_logger import save_trajectory_csv


def run_pid_simulation(
    duration: float = 10.0,
    dt: float = 0.004,
    roll_cmd: float = 0.0,
    pitch_cmd: float = 0.0,
    yaw_rate_cmd: float = 0.0,
    speed_cmd: float = 20.0,
    config: PIDConfig = None,
):
    """Run simulation with cascade PID controller.

    Args:
        duration: Simulation duration [s]
        dt: Timestep [s]
        roll_cmd: Target roll angle [deg]
        pitch_cmd: Target pitch angle [deg]
        yaw_rate_cmd: Target yaw rate [deg/s]
        speed_cmd: Target airspeed [m/s]

    Returns:
        Dictionary with simulation results
    """
    # Convert commands to radians
    setpoints = jnp.array([
        jnp.deg2rad(roll_cmd),
        jnp.deg2rad(pitch_cmd),
        jnp.deg2rad(yaw_rate_cmd),
        speed_cmd,
    ])
    roll_cmd_rad = setpoints[0]
    pitch_cmd_rad = setpoints[1]
    yaw_rate_cmd_rad = setpoints[2]

    # Initial aircraft state
    # Pos: [0,0,-100] (100m altitude), Vel: [20,0,0] (20m/s forward)
    # Quat: [1,0,0,0] (level), Omega: [0,0,0]
    state0 = jnp.array([
        0.0, 0.0, -100.0,  # Position (NED)
        20.0, 0.0, 0.0,     # Velocity (NED)
        1.0, 0.0, 0.0, 0.0, # Quaternion
        0.0, 0.0, 0.0,      # Angular velocity
        0.0, 0.0, 0.0, 0.0, # Actuator states
    ])

    # PID configuration and state
    if config is None:
        config = create_pid_config()
    pid_state0 = create_pid_state()

    steps = int(duration / dt)

    # Combined step function
    def step_fn(carry, _):
        state, pid_state = carry

        # Compute rate setpoints for logging
        quat = state[6:10]
        euler = quat_to_euler_jax(quat)
        roll = euler[0]
        pitch = euler[1]
        p_sp, q_sp = attitude_controller(roll_cmd_rad, pitch_cmd_rad, roll, pitch, config)
        r_sp = jnp.clip(yaw_rate_cmd_rad, -config.rate_limit, config.rate_limit)
        rate_sp = jnp.array([p_sp, q_sp, r_sp])
        rates = state[10:13]

        # PID controller
        actuators, new_pid_state = cascade_pid_step(
            setpoints, state, pid_state, config, dt
        )

        # Physics step
        next_state = equations_of_motion(state, actuators, dt)

        return (next_state, new_pid_state), (next_state, actuators, rate_sp, rates)

    # Run simulation
    (final_state, final_pid_state), (trajectory, actuators, rate_sp, rates) = jax.lax.scan(
        step_fn, (state0, pid_state0), jnp.arange(steps)
    )

    # Convert to numpy for plotting
    traj_np = np.array(trajectory)
    act_np = np.array(actuators)
    rate_sp_np = np.array(rate_sp)
    rates_np = np.array(rates)
    t = np.arange(steps) * dt

    # Extract Euler angles from quaternions
    euler_angles = []
    for i in range(steps):
        quat = traj_np[i, 6:10]
        euler = quat_to_euler_jax(jnp.array(quat))
        euler_angles.append(np.array(euler))
    euler_np = np.array(euler_angles)

    return {
        "t": t,
        "trajectory": traj_np,
        "actuators": act_np,
        "rate_sp": rate_sp_np,
        "rates": rates_np,
        "euler": euler_np,
        "setpoints": np.array(setpoints),
        "final_state": np.array(final_state),
    }


def plot_results(results, setpoints_deg, save_path="pid_test.png"):
    """Plot simulation results."""
    t = results["t"]
    euler = results["euler"]
    traj = results["trajectory"]
    act = results["actuators"]

    roll_cmd, pitch_cmd, yaw_rate_cmd, speed_cmd = setpoints_deg

    fig, axes = plt.subplots(4, 2, figsize=(14, 13))

    # Roll tracking
    ax = axes[0, 0]
    ax.plot(t, np.rad2deg(euler[:, 0]), label="Actual")
    ax.axhline(roll_cmd, color="r", linestyle="--", label="Setpoint")
    ax.set_ylabel("Roll (deg)")
    ax.set_title("Roll Tracking")
    ax.legend()
    ax.grid(True)

    # Pitch tracking
    ax = axes[0, 1]
    ax.plot(t, np.rad2deg(euler[:, 1]), label="Actual")
    ax.axhline(pitch_cmd, color="r", linestyle="--", label="Setpoint")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Pitch Tracking")
    ax.legend()
    ax.grid(True)

    # Speed tracking
    ax = axes[1, 0]
    speed = np.linalg.norm(traj[:, 3:6], axis=1)
    ax.plot(t, speed, label="Actual")
    ax.axhline(speed_cmd, color="r", linestyle="--", label="Setpoint")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Tracking")
    ax.legend()
    ax.grid(True)

    # Altitude
    ax = axes[1, 1]
    ax.plot(t, -traj[:, 2])  # NED: -z = altitude
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude")
    ax.grid(True)

    # Actuators
    ax = axes[2, 0]
    ax.plot(t, act[:, 0], label="Aileron")
    ax.plot(t, act[:, 1], label="Elevator")
    ax.plot(t, act[:, 2], label="Rudder")
    ax.set_ylabel("Deflection")
    ax.set_xlabel("Time (s)")
    ax.set_title("Control Surfaces")
    ax.legend()
    ax.grid(True)

    # Throttle
    ax = axes[2, 1]
    ax.plot(t, act[:, 3])
    ax.set_ylabel("Throttle")
    ax.set_xlabel("Time (s)")
    ax.set_title("Throttle")
    ax.grid(True)

    # Rate tracking (p, q)
    ax = axes[3, 0]
    rates = results["rates"]
    rate_sp = results["rate_sp"]
    ax.plot(t, np.rad2deg(rates[:, 0]), label="p")
    ax.plot(t, np.rad2deg(rate_sp[:, 0]), "--", label="p_sp")
    ax.plot(t, np.rad2deg(rates[:, 1]), label="q")
    ax.plot(t, np.rad2deg(rate_sp[:, 1]), "--", label="q_sp")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Rate Tracking (p, q)")
    ax.legend()
    ax.grid(True)

    # Rate tracking (r)
    ax = axes[3, 1]
    ax.plot(t, np.rad2deg(rates[:, 2]), label="r")
    ax.plot(t, np.rad2deg(rate_sp[:, 2]), "--", label="r_sp")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Rate Tracking (r)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Test Cascade PID controller")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (s)")
    parser.add_argument("--dt", type=float, default=0.004, help="Timestep (s)")
    parser.add_argument("--roll", type=float, default=30.0, help="Roll command (deg)")
    parser.add_argument("--pitch", type=float, default=-5.0, help="Pitch command (deg)")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw rate (deg/s)")
    parser.add_argument("--speed", type=float, default=20.0, help="Speed command (m/s)")
    parser.add_argument("--out", default="pid_test.png", help="Output plot path")
    parser.add_argument("--tuned", type=str, default=None,
                        help="Load tuned config from JSON file")
    args = parser.parse_args()

    # Load config
    config = None
    if args.tuned:
        try:
            config = load_tuned_config(args.tuned)
            print(f"Loaded tuned config from: {args.tuned}")
        except FileNotFoundError:
            print(f"Warning: {args.tuned} not found, using default config")

    print(f"Running PID test simulation...")
    print(f"  Roll cmd: {args.roll}°")
    print(f"  Pitch cmd: {args.pitch}°")
    print(f"  Yaw rate cmd: {args.yaw_rate}°/s")
    print(f"  Speed cmd: {args.speed} m/s")

    results = run_pid_simulation(
        duration=args.duration,
        dt=args.dt,
        roll_cmd=args.roll,
        pitch_cmd=args.pitch,
        yaw_rate_cmd=args.yaw_rate,
        speed_cmd=args.speed,
        config=config,
    )

    setpoints_deg = (args.roll, args.pitch, args.yaw_rate, args.speed)
    plot_results(results, setpoints_deg, save_path=args.out)

    # Print final state
    final_euler = results["euler"][-1]
    final_speed = np.linalg.norm(results["trajectory"][-1, 3:6])
    print(f"\nFinal state:")
    print(f"  Roll: {np.rad2deg(final_euler[0]):.1f}° (cmd: {args.roll}°)")
    print(f"  Pitch: {np.rad2deg(final_euler[1]):.1f}° (cmd: {args.pitch}°)")
    print(f"  Speed: {final_speed:.1f} m/s (cmd: {args.speed} m/s)")


if __name__ == "__main__":
    main()

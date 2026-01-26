"""CSV logging utilities for simulation trajectories."""

import numpy as np


def save_trajectory_csv(trajectory, dt, log_dt=0.1, filepath="sim_log.csv"):
    """Save simulation trajectory to CSV file.

    Args:
        trajectory: Array of shape (steps, 17) containing state history
        dt: Simulation timestep (seconds)
        log_dt: Logging interval (seconds), default 0.1s
        filepath: Output file path

    Returns:
        filepath: Path to saved file
    """
    traj_np = np.array(trajectory)

    # Subsample at log_dt intervals
    log_stride = max(1, int(round(log_dt / dt)))
    log_traj = traj_np[::log_stride]

    # Create time column
    log_time = (np.arange(log_traj.shape[0]) * log_stride * dt).reshape(-1, 1)

    # Combine time and state
    csv_data = np.concatenate([log_time, log_traj], axis=1)

    # Header
    header = "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,p,q,r,act_ail,act_ele,act_rud,act_thr"

    np.savetxt(filepath, csv_data, delimiter=",", header=header, comments="")

    return filepath


def save_rate_debug_csv(time, rate_sp, rates, actuators, filepath="rate_debug.csv"):
    """Save rate tuning debug data to CSV file.

    Args:
        time: Array of shape (steps,)
        rate_sp: Array of shape (steps,)
        rates: Array of shape (steps, 3) [p, q, r]
        actuators: Array of shape (steps, 4) [ail, ele, rud, thr]
        filepath: Output file path
    """
    time_np = np.array(time).reshape(-1, 1)
    rate_sp_np = np.array(rate_sp).reshape(-1, 1)
    rates_np = np.array(rates)
    actuators_np = np.array(actuators)

    csv_data = np.concatenate([time_np, rate_sp_np, rates_np, actuators_np], axis=1)
    header = "t,rate_sp,p,q,r,act_ail,act_ele,act_rud,act_thr"

    np.savetxt(filepath, csv_data, delimiter=",", header=header, comments="")
    return filepath

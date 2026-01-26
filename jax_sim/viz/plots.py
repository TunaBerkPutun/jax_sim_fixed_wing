"""Visualization and plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jax_sim.utils.quaternion import quat_to_euler, quat_to_rotmat


def load_log(path):
    """Load simulation log from CSV file.

    Args:
        path: Path to CSV file

    Returns:
        Structured numpy array with named columns
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def build_cone_body(length=2.0, radius=0.5, segments=12):
    """Build a cone mesh for 3D visualization.

    Args:
        length: Cone length (nose to base)
        radius: Base radius
        segments: Number of segments around the cone

    Returns:
        Array of face vertices
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    base = np.stack(
        [np.zeros_like(angles), radius * np.cos(angles), radius * np.sin(angles)],
        axis=1,
    )
    tip = np.array([length, 0.0, 0.0])

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([tip, base[j], base[i]])
    return np.array(faces)


def plot_summary(data, out_path="sim_summary.png"):
    """Generate summary plot of simulation data.

    Args:
        data: Structured array from load_log()
        out_path: Output file path

    Returns:
        Output file path
    """
    t = data["t"]
    x = data["px"]
    y = data["py"]
    z = data["pz"]
    vx = data["vx"]
    vy = data["vy"]
    vz = data["vz"]
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    roll, pitch, yaw = quat_to_euler(data["qw"], data["qx"], data["qy"], data["qz"])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, -z)
    ax1.set_title("Trajectory (X vs Altitude)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, y)
    ax2.set_title("Top View (X vs Y)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, speed, label="Speed")
    ax3.plot(t, np.degrees(roll), label="Roll (deg)")
    ax3.plot(t, np.degrees(pitch), label="Pitch (deg)")
    ax3.plot(t, np.degrees(yaw), label="Yaw (deg)")
    ax3.set_title("Speed + Attitude")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, data["act_ail"], label="Aileron")
    ax4.plot(t, data["act_ele"], label="Elevator")
    ax4.plot(t, data["act_rud"], label="Rudder")
    ax4.plot(t, data["act_thr"], label="Throttle")
    ax4.set_title("Actuators")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def animate_simple(data, out_path=None):
    """Create 2D animation of trajectory and speed.

    Args:
        data: Structured array from load_log()
        out_path: Output file path (None for interactive display)
    """
    t = data["t"]
    x = data["px"]
    y = data["py"]
    z = data["pz"]
    vx = data["vx"]
    vy = data["vy"]
    vz = data["vz"]
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    fig, (ax_traj, ax_speed) = plt.subplots(1, 2, figsize=(12, 5))

    ax_traj.plot(x, -z, color="#666666", linewidth=1.0)
    (point,) = ax_traj.plot([], [], "o", color="#1f77b4")
    ax_traj.set_title("Trajectory (X vs Altitude)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Altitude (m)")
    ax_traj.grid(True)

    ax_speed.plot(t, speed, color="#444444")
    cursor = ax_speed.axvline(t[0], color="#d62728", linewidth=1.5)
    ax_speed.set_title("Speed (m/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed")
    ax_speed.grid(True)

    ax_traj.set_xlim(np.min(x), np.max(x))
    ax_traj.set_ylim(np.min(-z), np.max(-z))
    ax_speed.set_xlim(np.min(t), np.max(t))
    ax_speed.set_ylim(0.0, max(1.0, np.max(speed)))

    def init():
        point.set_data([], [])
        cursor.set_xdata([t[0], t[0]])
        return point, cursor

    def update(i):
        point.set_data([x[i]], [-z[i]])
        cursor.set_xdata([t[i], t[i]])
        return point, cursor

    anim = FuncAnimation(
        fig, update, frames=len(t), init_func=init, interval=50, blit=True
    )

    if out_path:
        if out_path.endswith(".gif"):
            anim.save(out_path, writer="pillow", fps=20)
        else:
            anim.save(out_path, fps=20)
    else:
        plt.show()


def animate_3d(data, out_path=None):
    """Create 3D animation with cone body.

    Args:
        data: Structured array from load_log()
        out_path: Output file path (None for interactive display)
    """
    t = data["t"]
    x = data["px"]
    y = data["py"]
    z = data["pz"]

    cone_faces_body = build_cone_body(length=2.0, radius=0.5, segments=16)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Trajectory (Cone Body)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Path (Z up for visualization)
    ax.plot(x, y, -z, color="#666666", linewidth=1.0)

    cone = Poly3DCollection(
        [], facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.9
    )
    ax.add_collection3d(cone)

    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))
    zlim = (np.min(-z), np.max(-z))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect(
        [
            max(1.0, xlim[1] - xlim[0]),
            max(1.0, ylim[1] - ylim[0]),
            max(1.0, zlim[1] - zlim[0]),
        ]
    )

    def update(i):
        rot = quat_to_rotmat(data["qw"][i], data["qx"][i], data["qy"][i], data["qz"][i])
        pos = np.array([x[i], y[i], z[i]])
        faces = []
        for face in cone_faces_body:
            pts = (rot @ face.T).T + pos
            pts[:, 2] = -pts[:, 2]  # visualize with Z up
            faces.append(pts)
        cone.set_verts(faces)
        return (cone,)

    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
    fig._anim = anim

    if out_path:
        if out_path.endswith(".gif"):
            anim.save(out_path, writer="pillow", fps=20)
        else:
            anim.save(out_path, fps=20)
    else:
        plt.show()

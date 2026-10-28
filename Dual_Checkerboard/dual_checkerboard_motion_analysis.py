#!/usr/bin/env python3
"""Analyze 1-Hz board A/B poses from dual_checkerboard_distance.py.

Usage: python dual_checkerboard_motion_analysis.py <SESSION_DIR>
Writes 8 plots, analysis_data.csv, and analysis_summary.txt.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITY_THRESHOLD_PX = 5.0
COLORS = {"a": "tab:blue", "b": "tab:orange"}


def analyze(session_dir):
    session_dir = Path(session_dir)
    source = session_dir / "pose_1hz.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("No observations in pose_1hz.csv")
    df = df.sort_values("elapsed_s").reset_index(drop=True)

    required = ["elapsed_s", "tracking_epoch", "pair_distance_cm"]
    for k in ("a", "b"):
        required += [
            f"{k}_pose_valid", f"{k}_x_cm", f"{k}_y_cm", f"{k}_z_cm",
            f"{k}_distance_cm", f"{k}_roll_deg", f"{k}_pitch_deg",
            f"{k}_yaw_deg", f"{k}_reprojection_rmse_px",
            f"{k}_rvec_x", f"{k}_rvec_y", f"{k}_rvec_z",
        ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = {}
    good = {}
    for k in ("a", "b"):
        core = [f"{k}_{c}" for c in (
            "x_cm", "y_cm", "z_cm", "rvec_x", "rvec_y", "rvec_z")]
        valid[k] = df[f"{k}_pose_valid"].eq(1) & df[core].notna().all(axis=1)
        good[k] = valid[k] & df[f"{k}_reprojection_rmse_px"].le(QUALITY_THRESHOLD_PX)
        df[f"{k}_quality_flag"] = np.where(
            ~valid[k], "no_pose", np.where(good[k], "screened_ok", "high_error")
        )
        for col in ("vx_cm_s", "vy_cm_s", "vz_cm_s",
                    "speed_cm_s", "angular_speed_deg_s"):
            df[f"{k}_{col}"] = np.nan

    both_good = good["a"] & good["b"] & df["pair_distance_cm"].notna()
    df["pair_quality_flag"] = np.where(both_good, "screened_ok", "unavailable")
    df["pair_distance_screened_cm"] = df["pair_distance_cm"].where(both_good)
    df["pair_separation_rate_cm_s"] = np.nan

    # Velocity/angular speed are 1-Hz finite differences, ONLY for adjacent
    # logged frames where both poses of a given board pass quality screening.
    # Different tracking epochs must never be bridged after an R reset.
    for i in range(1, len(df)):
        dt = float(df.at[i, "elapsed_s"] - df.at[i-1, "elapsed_s"])
        if not (0 < dt <= 1.8):
            continue
        if df.at[i, "tracking_epoch"] != df.at[i-1, "tracking_epoch"]:
            continue

        for k in ("a", "b"):
            if not (good[k].iloc[i] and good[k].iloc[i-1]):
                continue
            cols = [f"{k}_{a}" for a in ("x_cm", "y_cm", "z_cm")]
            p0 = df.loc[i-1, cols].to_numpy(dtype=float)
            p1 = df.loc[i, cols].to_numpy(dtype=float)
            v = (p1-p0) / dt
            for suffix, val in zip(("vx_cm_s", "vy_cm_s", "vz_cm_s"), v):
                df.at[i, f"{k}_{suffix}"] = val
            df.at[i, f"{k}_speed_cm_s"] = float(np.linalg.norm(v))

            rc = [f"{k}_{a}" for a in ("rvec_x", "rvec_y", "rvec_z")]
            r0 = df.loc[i-1, rc].to_numpy(dtype=float)
            r1 = df.loc[i, rc].to_numpy(dtype=float)
            R0, _ = cv2.Rodrigues(r0)
            R1, _ = cv2.Rodrigues(r1)
            Rrel = R0.T @ R1
            cos_theta = np.clip((np.trace(Rrel)-1) / 2, -1, 1)
            df.at[i, f"{k}_angular_speed_deg_s"] = (
                float(np.degrees(np.arccos(cos_theta))) / dt
            )

        if both_good.iloc[i] and both_good.iloc[i-1]:
            df.at[i, "pair_separation_rate_cm_s"] = (
                df.at[i, "pair_distance_cm"] - df.at[i-1, "pair_distance_cm"]
            ) / dt

    df.to_csv(session_dir / "analysis_data.csv", index=False)
    plots = session_dir / "plots"
    plots.mkdir(exist_ok=True)
    t = df["elapsed_s"]

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)

    # 1: The actual three-distance goal
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ("a", "b"):
        ax.plot(t, df[f"{k}_distance_cm"].where(valid[k]), ".-",
                color=COLORS[k], label=f"Camera to {k.upper()}")
    ax.plot(t, df["pair_distance_cm"].where(valid["a"] & valid["b"]),
            ".-", color="tab:green", label="A to B")
    ax.set(title="Three checkerboard-center distances", xlabel="Time (s)",
           ylabel="Distance (cm)")
    ax.legend(); ax.grid(alpha=.3)
    finish(fig, "01_three_distances.png")

    # 2: X/Y/Z traces (each board in its own row)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, k in zip(axs, ("a", "b")):
        for component in ("x", "y", "z"):
            ax.plot(t, df[f"{k}_{component}_cm"].where(valid[k]), ".-",
                    label=component.upper())
        ax.set(ylabel="Position (cm)", title=f"Board {k.upper()} position")
        ax.grid(alpha=.3); ax.legend()
    axs[-1].set_xlabel("Time (s)")
    finish(fig, "02_xyz_positions.png")

    # 3: Euler angles for each board
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, k in zip(axs, ("a", "b")):
        for angle in ("roll", "pitch", "yaw"):
            ax.plot(t, df[f"{k}_{angle}_deg"].where(valid[k]), ".-",
                    label=angle.capitalize())
        ax.set(ylabel="Angle (deg)", title=f"Board {k.upper()} orientation (ZYX)")
        ax.grid(alpha=.3); ax.legend()
    axs[-1].set_xlabel("Time (s)")
    finish(fig, "03_roll_pitch_yaw.png")

    # 4: Shared 3D frame. Gaps deliberately remain missing.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], marker="^", c="black", s=65, label="Camera")
    for k in ("a", "b"):
        xyz = [df[f"{k}_{c}_cm"].where(valid[k]).to_numpy(dtype=float)
               for c in ("x", "y", "z")]
        ax.plot(*xyz, ".-", color=COLORS[k], label=f"Board {k.upper()}")
    ax.set(xlabel="X right (cm)", ylabel="Y down (cm)",
           zlabel="Z forward (cm)", title="3D board-center trajectories")
    ax.legend()
    finish(fig, "04_3d_trajectories.png")

    # 5: Translation speed
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ("a", "b"):
        ax.plot(t, df[f"{k}_speed_cm_s"], ".-", color=COLORS[k],
                label=f"Board {k.upper()}")
    ax.set(xlabel="Time (s)", ylabel="Speed (cm/s)",
           title="Estimated translational speeds (1 Hz)")
    ax.grid(alpha=.3); ax.legend()
    finish(fig, "05_translational_speeds.png")

    # 6: Angular speed
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ("a", "b"):
        ax.plot(t, df[f"{k}_angular_speed_deg_s"], ".-", color=COLORS[k],
                label=f"Board {k.upper()}")
    ax.set(xlabel="Time (s)", ylabel="Angular speed (deg/s)",
           title="Estimated angular speeds (1 Hz)")
    ax.grid(alpha=.3); ax.legend()
    finish(fig, "06_angular_speeds.png")

    # 7: Reprojection fit of PnP to image corners
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ("a", "b"):
        ax.plot(t, df[f"{k}_reprojection_rmse_px"].where(valid[k]), ".-",
                color=COLORS[k], label=f"Board {k.upper()}")
    ax.axhline(QUALITY_THRESHOLD_PX, color="gray", linestyle="--",
               label=f"Screening threshold ({QUALITY_THRESHOLD_PX:g} px)")
    ax.set(xlabel="Time (s)", ylabel="RMSE (px)",
           title="Pose reprojection error")
    ax.grid(alpha=.3); ax.legend()
    finish(fig, "07_reprojection_errors.png")

    # 8: Change in separation. Negative = moving closer (in these samples).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, df["pair_separation_rate_cm_s"], ".-", color="tab:green")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="d(separation)/dt (cm/s)",
           title="Rate of change of A-B separation (negative = approaching)")
    ax.grid(alpha=.3)
    finish(fig, "08_separation_rate.png")

    def median_fmt(series, units, decimals=1):
        x = series.dropna()
        return f"{x.median():.{decimals}f} {units}" if not x.empty else "N/A"

    pair = df.loc[valid["a"] & valid["b"], "pair_distance_cm"].dropna()
    lines = [
        f"Samples: {len(df)}",
        f"Board A valid poses: {int(valid['a'].sum())}",
        f"Board B valid poses: {int(valid['b'].sum())}",
        f"Both boards simultaneously valid: {int((valid['a'] & valid['b']).sum())}",
        f"Board A screened poses (<= {QUALITY_THRESHOLD_PX} px): {int(good['a'].sum())}",
        f"Board B screened poses (<= {QUALITY_THRESHOLD_PX} px): {int(good['b'].sum())}",
        f"Both simultaneously screened: {int(both_good.sum())}",
    ]
    if len(pair):
        lines.append(f"A-B distance range: {pair.min():.1f} - {pair.max():.1f} cm")
    else:
        lines.append("A-B distance range: N/A")
    for k in ("a", "b"):
        lines += [
            f"Board {k.upper()} median reprojection RMSE: "
            + median_fmt(df.loc[valid[k], f"{k}_reprojection_rmse_px"], "px", 2),
            f"Board {k.upper()} median translation speed: "
            + median_fmt(df[f"{k}_speed_cm_s"], "cm/s"),
            f"Board {k.upper()} median angular speed: "
            + median_fmt(df[f"{k}_angular_speed_deg_s"], "deg/s"),
        ]
    lines += [
        "",
        "Speeds are 1-Hz finite differences, not full-frame measurements.",
        "Missing/failed/poor-fit poses are not interpolated.",
        "R resets tracking IDs; speed estimates do not cross reset boundaries.",
        "Identical boards can swap IDs after crossing or occlusion.",
        "Reprojection error does not establish metric calibration accuracy.",
        "Distances are between checkerboard centers, not body/robot surfaces.",
    ]
    summary = "\n".join(lines) + "\n"
    (session_dir / "analysis_summary.txt").write_text(summary)
    print(summary)
    print("Plots saved to:", plots)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python dual_checkerboard_motion_analysis.py SESSION_DIR"
        )
    analyze(Path(sys.argv[1]))

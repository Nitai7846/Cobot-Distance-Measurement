#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:05:46 2026

@author: nitaishah
"""


import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

QUALITY_THRESHOLD_PX = 5.0


def analyze(session_dir):

    source = session_dir / "pose_1hz.csv"

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    df = df.sort_values(
        "elapsed_s"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("CSV contains no observations")

    plots = session_dir / "plots"
    plots.mkdir(exist_ok=True)

    # ======================================
    # CONVERT NUMERIC COLUMNS
    # ======================================

    numeric = [
        "elapsed_s",
        "x_cm", "y_cm", "z_cm",
        "distance_cm",
        "roll_deg", "pitch_deg", "yaw_deg",
        "reprojection_rmse_px",
        "rvec_x", "rvec_y", "rvec_z"
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # ======================================
    # DETERMINE VALID POSES
    # ======================================

    valid = (
        df["pose_valid"].fillna(0).eq(1)
        & df[
            [
                "x_cm", "y_cm", "z_cm",
                "rvec_x", "rvec_y", "rvec_z"
            ]
        ].notna().all(axis=1)
    )

    good = (
        valid
        & df["reprojection_rmse_px"].le(
            QUALITY_THRESHOLD_PX
        )
    )

    df["quality_flag"] = np.where(
        ~valid,
        "no_pose",
        np.where(
            good,
            "screened_ok",
            "high_error"
        )
    )

    # ======================================
    # INITIALIZE VELOCITY COLUMNS
    # ======================================

    velocity_columns = [
        "vx_cm_s",
        "vy_cm_s",
        "vz_cm_s",
        "speed_cm_s",
        "angular_speed_deg_s"
    ]

    for col in velocity_columns:
        df[col] = np.nan

    # ======================================
    # COMPUTE TRANSLATIONAL VELOCITY
    # AND ANGULAR SPEED
    # ======================================

    for i in range(1, len(df)):

        # Only consecutive reliable detections
        if not (good.iloc[i] and good.iloc[i - 1]):
            continue

        dt = (
            df.at[i, "elapsed_s"]
            - df.at[i - 1, "elapsed_s"]
        )

        # Do not bridge detection gaps
        if not (0 < dt <= 1.8):
            continue

        p0 = df.loc[
            i - 1, ["x_cm", "y_cm", "z_cm"]
        ].to_numpy(dtype=float)

        p1 = df.loc[
            i, ["x_cm", "y_cm", "z_cm"]
        ].to_numpy(dtype=float)

        velocity = (p1 - p0) / dt

        df.loc[
            i, ["vx_cm_s", "vy_cm_s", "vz_cm_s"]
        ] = velocity

        df.at[i, "speed_cm_s"] = float(
            np.linalg.norm(velocity)
        )

        # ----------------------------------
        # COMPUTE RELATIVE ROTATION
        # ----------------------------------

        r0 = df.loc[
            i - 1,
            ["rvec_x", "rvec_y", "rvec_z"]
        ].to_numpy(dtype=float)

        r1 = df.loc[
            i,
            ["rvec_x", "rvec_y", "rvec_z"]
        ].to_numpy(dtype=float)

        R0, _ = cv2.Rodrigues(r0)
        R1, _ = cv2.Rodrigues(r1)

        R_relative = R0.T @ R1

        cos_angle = np.clip(
            (np.trace(R_relative) - 1.0) / 2.0,
            -1.0,
            1.0
        )

        angular_speed = (
            np.degrees(np.arccos(cos_angle)) / dt
        )

        df.at[
            i, "angular_speed_deg_s"
        ] = float(angular_speed)

    # ======================================
    # SAVE ANALYZED CSV
    # ======================================

    df.to_csv(
        session_dir / "analysis_data.csv",
        index=False
    )

    t = df["elapsed_s"]

    # ======================================
    # REUSABLE PLOTTING FUNCTION
    # ======================================

    def line_plot(columns, ylabel, filename, title):

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        for col, label in columns:

            if col in (
                "speed_cm_s",
                "angular_speed_deg_s"
            ):
                values = df[col]
            else:
                values = df[col].where(valid)

            ax.plot(
                t,
                values,
                marker=".",
                label=label
            )

        ax.set(
            title=title,
            xlabel="Elapsed time (s)",
            ylabel=ylabel
        )

        ax.grid(alpha=0.3)

        if len(columns) > 1:
            ax.legend()

        fig.tight_layout()

        fig.savefig(
            plots / filename,
            dpi=160
        )

        plt.close(fig)

    # ======================================
    # PLOT 1: DISTANCE VS TIME
    # ======================================

    line_plot(
        [("distance_cm", "Distance")],
        "Distance (cm)",
        "01_distance_vs_time.png",
        "Distance to checkerboard center"
    )

    # ======================================
    # PLOT 2: X, Y, Z VS TIME
    # ======================================

    line_plot(
        [
            ("x_cm", "X"),
            ("y_cm", "Y"),
            ("z_cm", "Z")
        ],
        "Position (cm)",
        "02_xyz_position.png",
        "3D position vs. time"
    )

    # ======================================
    # PLOT 3: ROLL, PITCH, YAW
    # ======================================

    line_plot(
        [
            ("roll_deg", "Roll"),
            ("pitch_deg", "Pitch"),
            ("yaw_deg", "Yaw")
        ],
        "Angle (degrees)",
        "03_roll_pitch_yaw.png",
        "Orientation vs. time (ZYX Euler angles)"
    )

    # ======================================
    # PLOT 4: 3D TRAJECTORY
    # ======================================

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111, projection="3d"
    )

    x, y, z = (
        df[c].where(valid).to_numpy(dtype=float)
        for c in ("x_cm", "y_cm", "z_cm")
    )

    ax.plot(
        x, y, z,
        marker="o",
        markersize=3
    )

    # Camera at coordinate origin
    ax.scatter(
        [0], [0], [0],
        marker="^",
        s=60,
        label="Camera"
    )

    ax.set(
        xlabel="X (cm) — right",
        ylabel="Y (cm) — down",
        zlabel="Z (cm) — forward",
        title="Checkerboard center trajectory"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        plots / "04_3d_trajectory.png",
        dpi=160
    )

    plt.close(fig)

    # ======================================
    # PLOT 5: TRANSLATIONAL SPEED
    # ======================================

    line_plot(
        [("speed_cm_s", "Speed")],
        "Speed (cm/s)",
        "05_translational_speed.png",
        "Estimated translational speed"
    )

    # ======================================
    # PLOT 6: ANGULAR SPEED
    # ======================================

    line_plot(
        [
            ("angular_speed_deg_s", "Angular speed")
        ],
        "Angular speed (deg/s)",
        "06_angular_speed.png",
        "Estimated angular speed"
    )

    # ======================================
    # PLOT 7: REPROJECTION ERROR
    # ======================================

    line_plot(
        [
            (
                "reprojection_rmse_px",
                "Reprojection error"
            )
        ],
        "Error (pixels)",
        "07_reprojection_error.png",
        "Pose reprojection RMSE"
    )

    # ======================================
    # SAVE ANALYSIS SUMMARY
    # ======================================

    reproj = df.loc[
        valid, "reprojection_rmse_px"
    ]

    summary = (
        f"Samples: {len(df)}\n"
        f"Valid poses: {int(valid.sum())}\n"
        f"Screened poses: {int(good.sum())}\n"
        f"Median reprojection error: "
        f"{reproj.median():.2f} px\n"
        f"Distance range: "
        f"{df.loc[valid, 'distance_cm'].min():.1f} - "
        f"{df.loc[valid, 'distance_cm'].max():.1f} cm\n"
        f"Median translational speed: "
        f"{df['speed_cm_s'].median():.1f} cm/s\n"
        f"Median angular speed: "
        f"{df['angular_speed_deg_s'].median():.1f} deg/s\n"
        "\nSpeeds are 1-Hz estimates.\n"
        "Missing/flagged poses are not interpolated.\n"
        "Reprojection filtering does not guarantee accuracy.\n"
    )

    (
        session_dir / "analysis_summary.txt"
    ).write_text(summary)

    print(summary)
    print("Plots saved to:", plots)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python checkerboard_motion_analysis.py "
            "SESSION_FOLDER"
        )

    analyze(Path(sys.argv[1]))

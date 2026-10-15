#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:42:30 2026

@author: nitaishah
"""


import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

matplotlib.rcParams[
    "animation.ffmpeg_path"
] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import pandas as pd


# ==========================================
# VIEW CONFIGURATIONS
# ==========================================

# Format:
# (filename, axis_mode, elevation, azimuth, title)

VIEWS = [
    ("01_original_z_vertical", "native", 30, -60,
     "Original: Z vertical"),

    ("02_original_from_opposite", "native", 30, 120,
     "Z vertical: opposite side"),

    ("03_upright_isometric_a", "upright", 25, -65,
     "Upright 3D: diagonal A"),

    ("04_upright_isometric_b", "upright", 25, 25,
     "Upright 3D: diagonal B"),

    ("05_upright_isometric_c", "upright", 25, 115,
     "Upright 3D: diagonal C"),

    ("06_upright_isometric_d", "upright", 25, 205,
     "Upright 3D: diagonal D"),

    ("07_upright_from_camera", "upright", 0, -90,
     "Looking from camera"),

    ("08_upright_from_behind", "upright", 0, 90,
     "Looking toward camera"),

    ("09_upright_from_left", "upright", 0, 180,
     "Left-side profile"),

    ("10_upright_from_right", "upright", 0, 0,
     "Right-side profile"),

    ("11_upright_birds_eye", "upright", 65, -65,
     "High-angle view"),

    ("12_upright_top_down", "upright", 90, -90,
     "Directly overhead"),
]

AXIS_LABELS = {
    "native": (
        "X (cm) - right",
        "Y (cm) - down",
        "Z (cm) - forward"
    ),
    "upright": (
        "X (cm) - right",
        "Depth Z (cm) - forward",
        "Height -Y (cm) - up"
    )
}


# ==========================================
# AXIS TRANSFORMATION
# ==========================================

def transform(points, mode):

    points = np.asarray(points, dtype=float)

    if mode == "native":
        return points

    # Original OpenCV coordinates:
    # X = right
    # Y = down
    # Z = forward
    #
    # Upright visualization:
    # plot X = camera X
    # plot Y = camera Z (depth)
    # plot Z = -camera Y (height)

    return (
        points[..., [0, 2, 1]]
        * np.array([1.0, 1.0, -1.0])
    )


# ==========================================
# LOAD RECORDED POSES
# ==========================================

def load_recording(session):

    df = pd.read_csv(
        session / "pose_1hz.csv"
    )

    df = df.sort_values(
        "elapsed_s"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No recorded observations")

    metadata = json.loads(
        (session / "metadata.json").read_text()
    )

    nx, ny = metadata[
        "checkerboard_inner_corners"
    ]

    size = float(
        metadata["square_size_mm"]
    )

    # Physical checkerboard outer corners (mm)
    board_corners = np.array([
        [-size, -size, 0],
        [nx * size, -size, 0],
        [nx * size, ny * size, 0],
        [-size, ny * size, 0]
    ], dtype=float)

    required = [
        "x_cm", "y_cm", "z_cm",
        "rvec_x", "rvec_y", "rvec_z",
        "tx_mm", "ty_mm", "tz_mm",
        "elapsed_s"
    ]

    for col in required + ["pose_valid"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    valid = (
        df["pose_valid"].fillna(0).eq(1)
        & df[required].notna().all(axis=1)
    )

    if not valid.any():
        raise ValueError("No valid poses to animate")

    centers = np.full(
        (len(df), 3), np.nan
    )

    quads = np.full(
        (len(df), 4, 3), np.nan
    )

    # Reconstruct board position and orientation
    for i in np.flatnonzero(valid.to_numpy()):

        rvec = df.loc[
            i,
            ["rvec_x", "rvec_y", "rvec_z"]
        ].to_numpy(dtype=float)

        tvec = df.loc[
            i,
            ["tx_mm", "ty_mm", "tz_mm"]
        ].to_numpy(dtype=float)

        R, _ = cv2.Rodrigues(rvec)

        quads[i] = (
            board_corners @ R.T + tvec
        ) / 10.0

        centers[i] = df.loc[
            i,
            ["x_cm", "y_cm", "z_cm"]
        ].to_numpy(dtype=float)

    return df, valid.to_numpy(), centers, quads


# ==========================================
# FIXED AXIS LIMITS
# ==========================================

def get_bounds(centers, quads, mode):

    center_valid = centers[
        np.isfinite(centers).all(axis=1)
    ]

    quad_valid = quads[
        np.isfinite(quads).all(axis=(1, 2))
    ].reshape(-1, 3)

    points = np.concatenate([
        np.zeros((1, 3)),
        center_valid,
        quad_valid
    ])

    points = transform(points, mode)

    low = points.min(axis=0)
    high = points.max(axis=0)

    margin = np.maximum(
        (high - low) * 0.08,
        5.0
    )

    return low - margin, high + margin


# ==========================================
# DRAW ONE FRAME
# ==========================================

def render(ax, i, view, df, valid,
           centers, quads, bounds):

    name, mode, elev, azim, label = view

    low, high = bounds

    ax.cla()

    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])

    ax.set_box_aspect(
        np.maximum(high - low, 1.0)
    )

    # This controls the 3D viewing angle
    ax.view_init(
        elev=elev,
        azim=azim
    )

    ax.set_xlabel(AXIS_LABELS[mode][0])
    ax.set_ylabel(AXIS_LABELS[mode][1])
    ax.set_zlabel(AXIS_LABELS[mode][2])

    # Camera position
    ax.scatter(
        [0], [0], [0],
        c="black",
        marker="^",
        s=35,
        label="Webcam"
    )

    # Draw trajectory until current timestamp
    path = transform(
        centers[:i + 1],
        mode
    )

    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        linewidth=1.7,
        label="Center path"
    )

    if valid[i]:

        quad = transform(
            quads[i],
            mode
        )

        # Draw flat checkerboard plane
        board = Poly3DCollection(
            [quad],
            facecolors="tab:cyan",
            edgecolors="black",
            alpha=0.55
        )

        ax.add_collection3d(board)

        p = transform(
            centers[i],
            mode
        )

        ax.scatter(
            *p,
            color="tab:blue",
            s=32
        )

        status = "Pose detected"

    else:
        status = "NO POSE at this time"

    timestamp = df.at[i, "elapsed_s"]

    ax.set_title(
        f"{label} | t={timestamp:.1f}s\n{status}"
    )

    ax.legend(
        loc="upper left",
        fontsize=8
    )


# ==========================================
# GENERATE ALL 12 ANIMATIONS
# ==========================================

def animate(session, selected=None):

    session = Path(session)

    df, valid, centers, quads = load_recording(
        session
    )

    output_dir = session / "axis_views"
    output_dir.mkdir(exist_ok=True)

    configs = [
        view for view in VIEWS
        if selected is None
        or view[0].split("_")[0] in selected
    ]

    if not configs:
        raise ValueError("No matching view numbers")

    # Calculate fixed bounds for both mappings
    bounds = {
        mode: get_bounds(centers, quads, mode)
        for mode in ("native", "upright")
    }

    # ======================================
    # GENERATE COMPARISON IMAGE
    # ======================================

    indices = np.flatnonzero(valid)

    preview_index = indices[
        len(indices) // 2
    ]

    fig = plt.figure(
        figsize=(20, 15)
    )

    for j, view in enumerate(VIEWS):

        ax = fig.add_subplot(
            3, 4, j + 1,
            projection="3d"
        )

        render(
            ax,
            preview_index,
            view,
            df,
            valid,
            centers,
            quads,
            bounds[view[1]]
        )

        ax.set_title(
            view[0].replace("_", " "),
            fontsize=11
        )

        ax.legend().remove()

        ax.tick_params(
            labelsize=6,
            pad=0
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    fig.suptitle(
        "12 Axis Orientations and Viewing Angles",
        fontsize=18
    )

    fig.savefig(
        output_dir / "view_comparison.png",
        dpi=130
    )

    plt.close(fig)

    # ======================================
    # SAVE VIEW DESCRIPTIONS
    # ======================================

    descriptions = [
        "All views use the same estimated poses.",
        "Native axes: X right, Y down, Z forward.",
        "Upright: X right, depth Z forward, height -Y up.",
        ""
    ]

    for view in VIEWS:

        descriptions.append(
            f"{view[0]}: {view[4]} "
            f"(elev={view[2]}, azim={view[3]})"
        )

    (output_dir / "view_guide.txt").write_text(
        "\n".join(descriptions)
    )

    # ======================================
    # EXPORT VIDEOS
    # ======================================

    print(
        f"Generating {len(configs)} "
        f"animation variants..."
    )

    for j, view in enumerate(configs, start=1):

        name, mode, elev, azim, label = view

        target = output_dir / f"{name}.mp4"

        fig = plt.figure(
            figsize=(9, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        def frame_function(i):

            render(
                ax,
                i,
                view,
                df,
                valid,
                centers,
                quads,
                bounds[mode]
            )

            return []

        animation = FuncAnimation(
            fig,
            frame_function,
            frames=len(df),
            interval=1000,
            repeat=False,
            blit=False
        )

        print(
            f"[{j}/{len(configs)}] "
            f"Rendering {name}...",
            flush=True
        )

        animation.save(
            str(target),
            writer=FFMpegWriter(
                fps=1,
                bitrate=1800
            ),
            dpi=110
        )

        plt.close(fig)

        # Also keep the original familiar filename
        if name == "01_original_z_vertical":

            shutil.copyfile(
                target,
                session / "checkerboard_3d_animation.mp4"
            )

    print("\nAll animations saved!")
    print("Folder:", output_dir)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session_dir",
        type=Path
    )

    parser.add_argument(
        "--only",
        default=None,
        help="Optional view numbers, e.g. 03,07,11"
    )

    args = parser.parse_args()

    selected = (
        set(args.only.split(","))
        if args.only
        else None
    )

    animate(
        args.session_dir,
        selected
    )

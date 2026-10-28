#!/usr/bin/env python3
"""3D animation of two checkerboards and their estimated separation.

Usage: python dual_checkerboard_3d_animation.py SESSION_DIR [--only 03,07]

Uses the exact recorded 1-Hz timestamps and board-to-camera rvec/tvec.
No missing poses are interpolated. Twelve original camera viewpoints are retained.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

VIEWS = [
    ("01_original_z_vertical", "native", 30, -60, "Original: Z vertical"),
    ("02_original_from_opposite", "native", 30, 120, "Z vertical: opposite"),
    ("03_upright_isometric_a", "upright", 25, -65, "Upright diagonal A"),
    ("04_upright_isometric_b", "upright", 25, 25, "Upright diagonal B"),
    ("05_upright_isometric_c", "upright", 25, 115, "Upright diagonal C"),
    ("06_upright_isometric_d", "upright", 25, 205, "Upright diagonal D"),
    ("07_upright_from_camera", "upright", 0, -90, "Looking from camera"),
    ("08_upright_from_behind", "upright", 0, 90, "Looking toward camera"),
    ("09_upright_from_left", "upright", 0, 180, "Left profile"),
    ("10_upright_from_right", "upright", 0, 0, "Right profile"),
    ("11_upright_birds_eye", "upright", 65, -65, "High-angle view"),
    ("12_upright_top_down", "upright", 90, -90, "Directly overhead"),
]

AXIS_LABELS = {
    "native": ("X (cm) right", "Y (cm) down", "Z (cm) forward"),
    "upright": ("X (cm) right", "Depth Z (cm) forward", "Height -Y (cm) up"),
}
BOARD_COLORS = {"a": "tab:blue", "b": "tab:orange"}


def transform(points, mode):
    """Change only PLOT axes. Real poses remain in OpenCV camera axes."""
    pts = np.asarray(points, dtype=float)
    if mode == "native":
        return pts
    return pts[..., [0, 2, 1]] * [1.0, 1.0, -1.0]


def load_recording(session):
    df = pd.read_csv(session / "pose_1hz.csv")
    if df.empty:
        raise ValueError("No logged observations")
    df = df.sort_values("elapsed_s").reset_index(drop=True)
    meta = json.loads((session / "metadata.json").read_text())
    nx, ny = meta["checkerboard_inner_corners"]
    square = float(meta["square_size_mm"])
    # Physical board outer corners, in mm, relative to FIRST INNER corner.
    board_corners_mm = np.array([
        [-square, -square, 0.0],
        [nx * square, -square, 0.0],
        [nx * square, ny * square, 0.0],
        [-square, ny * square, 0.0],
    ])

    centers, quads, valid = {}, {}, {}
    n = len(df)
    for k in ("a", "b"):
        names = [f"{k}_{x}" for x in (
            "x_cm", "y_cm", "z_cm", "rvec_x", "rvec_y", "rvec_z",
            "tx_mm", "ty_mm", "tz_mm", "pose_valid")]
        missing = [x for x in names if x not in df.columns]
        if missing:
            raise ValueError(f"Missing CSV fields: {missing}")
        for name in names:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        valid[k] = (df[f"{k}_pose_valid"].eq(1)
                    & df[names].notna().all(axis=1)).to_numpy()
        centers[k] = np.full((n, 3), np.nan, dtype=float)
        quads[k] = np.full((n, 4, 3), np.nan, dtype=float)
        for i in np.flatnonzero(valid[k]):
            rvec = df.loc[i, [f"{k}_rvec_{d}" for d in "xyz"]].to_numpy(dtype=float)
            tvec = df.loc[i, [f"{k}_t{d}_mm" for d in "xyz"]].to_numpy(dtype=float)
            R, _ = cv2.Rodrigues(rvec)
            quads[k][i] = (board_corners_mm @ R.T + tvec) / 10.0
            centers[k][i] = df.loc[i, [f"{k}_{d}_cm" for d in "xyz"]].to_numpy(dtype=float)

    if not (valid["a"].any() or valid["b"].any()):
        raise ValueError("No valid checkerboard poses to animate")
    epochs = (pd.to_numeric(df["tracking_epoch"], errors="coerce")
              if "tracking_epoch" in df else pd.Series(np.zeros(n)))
    return df, centers, quads, valid, epochs.fillna(0).to_numpy()


def get_bounds(centers, quads, mode):
    arrays = [np.zeros((1, 3))]
    for k in ("a", "b"):
        pc = centers[k].reshape(-1, 3)
        pq = quads[k].reshape(-1, 3)
        arrays.extend([pc[np.isfinite(pc).all(axis=1)],
                       pq[np.isfinite(pq).all(axis=1)]])
    points = transform(np.concatenate(arrays), mode)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((high-low)*0.10, 5.0)
    return low-margin, high+margin


def render(ax, i, view, df, centers, quads, valid, epochs, bounds):
    filename, mode, elev, azim, label = view
    lo, hi = bounds
    ax.cla()
    ax.set_xlim(*[lo[0], hi[0]])
    ax.set_ylim(*[lo[1], hi[1]])
    ax.set_zlim(*[lo[2], hi[2]])
    ax.set_box_aspect(np.maximum(hi-lo, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(AXIS_LABELS[mode][0])
    ax.set_ylabel(AXIS_LABELS[mode][1])
    ax.set_zlabel(AXIS_LABELS[mode][2])
    ax.scatter([0], [0], [0], c="black", marker="^", s=40, label="Camera")

    for k in ("a", "b"):
        # Break trajectories on ID reset and missing detections.
        path = centers[k][:i+1].copy()
        # Each epoch is a separate curve; do not concatenate swapped labels.
        seen_epochs = np.unique(epochs[:i+1])
        first_segment = True
        for epoch in seen_epochs:
            subset = path.copy()
            subset[epochs[:i+1] != epoch] = np.nan
            trace = transform(subset, mode)
            ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                    color=BOARD_COLORS[k], linewidth=1.5,
                    label=f"Board {k.upper()} path" if first_segment else "_nolegend_")
            first_segment = False

        if valid[k][i]:
            quad = transform(quads[k][i], mode)
            board = Poly3DCollection([quad], facecolors=BOARD_COLORS[k],
                                     edgecolors="black", alpha=.4)
            ax.add_collection3d(board)
            p = transform(centers[k][i], mode)
            ax.scatter(*p, c=BOARD_COLORS[k], s=35)
            ax.text(*p, k.upper(), color=BOARD_COLORS[k], weight="bold")

    both = valid["a"][i] and valid["b"][i]
    if both:
        a = transform(centers["a"][i], mode)
        b = transform(centers["b"][i], mode)
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                "--", linewidth=2, color="tab:green", label="A-B separation")
        gap = float(np.linalg.norm(centers["a"][i] - centers["b"][i]))
        gap_text = f"A-B: {gap:.1f} cm"
    else:
        gap_text = "A-B unavailable (both required)"

    timestamp = float(df.at[i, "elapsed_s"])
    status_a = "seen" if valid["a"][i] else "missing"
    status_b = "seen" if valid["b"][i] else "missing"
    ax.set_title(f"{label} | t={timestamp:.1f}s | {gap_text}\n"
                 f"A: {status_a} | B: {status_b}", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)


def animate(session, selected=None):
    session = Path(session)
    df, centers, quads, valid, epochs = load_recording(session)
    out = session / "axis_views"
    out.mkdir(exist_ok=True)
    configs = [v for v in VIEWS if selected is None or v[0][:2] in selected]
    if not configs:
        raise ValueError("No view matched --only (use numbers 01 through 12)")
    bounds = {m: get_bounds(centers, quads, m) for m in ("native", "upright")}

    # Comparison image uses a timestamp where BOTH were seen if possible.
    both = np.flatnonzero(valid["a"] & valid["b"])
    available = both if len(both) else np.flatnonzero(valid["a"] | valid["b"])
    preview_index = int(available[len(available)//2])
    fig = plt.figure(figsize=(20, 15))
    for j, view in enumerate(VIEWS):
        ax = fig.add_subplot(3, 4, j+1, projection="3d")
        render(ax, preview_index, view, df, centers, quads, valid, epochs,
               bounds[view[1]])
        ax.set_title(view[0].replace("_", " "), fontsize=10)
        ax.get_legend().remove()
        ax.tick_params(labelsize=6, pad=0)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    fig.suptitle("Dual-checkerboard 3D viewing angles", fontsize=18)
    fig.savefig(out / "view_comparison.png", dpi=125)
    plt.close(fig)

    guides = [
        "All 12 views use the same measured rvec/tvec, with no interpolation.",
        "Native: X right, Y down, Z forward.",
        "Upright: X right, depth Z forward, height -Y up.",
        "A/B identities may switch when identical boards cross or are occluded.",
        "Only the board-center separation is estimated (not nearest surfaces).",
        "",
    ]
    guides += [f"{v[0]}: {v[4]} (elev={v[2]}, azim={v[3]})" for v in VIEWS]
    (out / "view_guide.txt").write_text("\n".join(guides) + "\n")

    print(f"Generating {len(configs)} animation view(s)...", flush=True)
    for idx, view in enumerate(configs, 1):
        name, mode, _, _, _ = view
        target = out / f"{name}.mp4"
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Local function with current view bound, used by FuncAnimation.
        def draw_frame(i):
            render(ax, i, view, df, centers, quads, valid, epochs, bounds[mode])
            return []

        animation = FuncAnimation(
            fig, draw_frame, frames=len(df), interval=1000,
            repeat=False, blit=False
        )
        print(f"[{idx}/{len(configs)}] Rendering {name}...", flush=True)
        animation.save(str(target), writer=FFMpegWriter(fps=1, bitrate=1800), dpi=110)
        plt.close(fig)
        if name == "01_original_z_vertical":
            shutil.copyfile(target, session / "dual_checkerboard_3d_animation.mp4")
    print("Animations saved to:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--only", default=None,
                        help="Comma-separated view numbers, e.g. 03,07,11")
    args = parser.parse_args()
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    animate(args.session_dir, only)

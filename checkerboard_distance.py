#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:05:20 2026

@author: nitaishah
"""


import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

CHECKERBOARD = (10, 7)  # Inner corners
SQUARE_SIZE_MM = 25.0

BASE_DIR = Path(__file__).resolve().parent
CALIB_PATH = BASE_DIR / "webcam_calibration.npz"

SESSION_DIR = (
    BASE_DIR
    / "Checkerboard_Results"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)

SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD CAMERA CALIBRATION
# ==========================================

if not CALIB_PATH.exists():
    raise FileNotFoundError(
        f"Run camera_calibration.py first: {CALIB_PATH}"
    )

with np.load(CALIB_PATH) as saved:
    K = saved["K"]
    dist = saved["dist"]
    calibrated_size = tuple(saved["image_size"].tolist())

    calibration_rms = (
        float(saved["rms"])
        if "rms" in saved.files
        else None
    )

# ==========================================
# GENERATE 3D CHECKERBOARD POINTS
# ==========================================

object_points = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2) * SQUARE_SIZE_MM
)

# Physical center relative to first inner corner
center_board = np.array([
    (CHECKERBOARD[0] - 1) * SQUARE_SIZE_MM / 2,
    (CHECKERBOARD[1] - 1) * SQUARE_SIZE_MM / 2,
    0.0
], dtype=np.float64).reshape(3, 1)

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================================
# ROTATION MATRIX TO EULER ANGLES
# ==========================================

def euler_zyx_degrees(R):

    sy = np.hypot(R[0, 0], R[1, 0])

    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


# ==========================================
# CSV COLUMNS
# ==========================================

fields = [
    "timestamp",
    "elapsed_s",
    "frame_index",
    "corners_detected",
    "pose_valid",
    "x_cm",
    "y_cm",
    "z_cm",
    "distance_cm",
    "tx_mm",
    "ty_mm",
    "tz_mm",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "center_u_px",
    "center_v_px",
    "reprojection_rmse_px"
]


# ==========================================
# RECORDING FUNCTION
# ==========================================

def record():

    cap = cv2.VideoCapture(
        0, cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    video = None

    try:

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            calibrated_size[0]
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            calibrated_size[1]
        )

        cap.set(cv2.CAP_PROP_FPS, 30)

        ok, first = cap.read()

        if not ok:
            raise RuntimeError("Could not read camera")

        height, width = first.shape[:2]

        if (width, height) != calibrated_size:
            raise RuntimeError(
                f"Camera is {width}x{height}; "
                f"calibration is {calibrated_size}"
            )

        capture_fps = cap.get(cv2.CAP_PROP_FPS)

        if (
            not np.isfinite(capture_fps)
            or capture_fps <= 0
        ):
            capture_fps = 30.0

        # ==================================
        # INITIALIZE VIDEO WRITER
        # ==================================

        video = cv2.VideoWriter(
            str(SESSION_DIR / "recording.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            capture_fps,
            (width, height)
        )

        if not video.isOpened():
            raise RuntimeError(
                "Could not initialize MP4 writer"
            )

        # ==================================
        # SAVE METADATA
        # ==================================

        metadata = {
            "session": SESSION_DIR.name,
            "checkerboard_inner_corners": list(CHECKERBOARD),
            "square_size_mm": SQUARE_SIZE_MM,
            "camera_matrix": K.tolist(),
            "distortion_coefficients": dist.ravel().tolist(),
            "calibration_rms_px": calibration_rms,
            "image_size": [width, height],
            "video_nominal_fps": capture_fps,
            "logging_interval_s": 1.0,
            "pose_frame": (
                "board origin = first inner corner; "
                "board-to-camera"
            ),
            "center_frame": (
                "physical checkerboard center "
                "in camera coordinates"
            ),
            "euler_convention": (
                "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"
            )
        }

        (SESSION_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        print("Recording started!")
        print("Press Q to stop.")
        print("Saving to:", SESSION_DIR)

        if calibration_rms is not None and calibration_rms > 2:
            print(
                f"CAUTION: Calibration RMS = "
                f"{calibration_rms:.2f} px"
            )

        start = time.monotonic()
        next_log_time = 0.0
        frame_index = 0

        # ==================================
        # MAIN RECORDING LOOP
        # ==================================

        with (
            (SESSION_DIR / "pose_1hz.csv").open(
                "w", newline=""
            ) as csv_file,
            (SESSION_DIR / "observations_1hz.jsonl").open(
                "w"
            ) as json_file
        ):

            writer = csv.DictWriter(
                csv_file, fieldnames=fields
            )

            writer.writeheader()

            while True:

                ok, frame = cap.read()

                if not ok:
                    print("Camera frame unavailable")
                    break

                frame_index += 1

                elapsed = time.monotonic() - start

                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )

                # --------------------------
                # DETECT CHECKERBOARD
                # --------------------------

                found, corners = cv2.findChessboardCorners(
                    gray,
                    CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH |
                    cv2.CALIB_CB_NORMALIZE_IMAGE
                )

                pose = None
                rotation_matrix = None
                corner_pixels = None

                if found:

                    corners = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        criteria
                    )

                    corner_pixels = (
                        corners.reshape(-1, 2).tolist()
                    )

                    # ----------------------
                    # ESTIMATE 6-DOF POSE
                    # ----------------------

                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        corners,
                        K,
                        dist,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success and tvec[2, 0] > 0:

                        R, _ = cv2.Rodrigues(rvec)

                        rotation_matrix = R.tolist()

                        center_cam = (
                            R @ center_board + tvec
                        )

                        x, y, z = center_cam.ravel()

                        roll, pitch, yaw = (
                            euler_zyx_degrees(R)
                        )

                        # ------------------
                        # REPROJECTION ERROR
                        # ------------------

                        reprojected, _ = cv2.projectPoints(
                            object_points,
                            rvec,
                            tvec,
                            K,
                            dist
                        )

                        error = (
                            reprojected.reshape(-1, 2)
                            - corners.reshape(-1, 2)
                        )

                        reproj_rmse = float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        error * error,
                                        axis=1
                                    )
                                )
                            )
                        )

                        # Project board center
                        image_center, _ = cv2.projectPoints(
                            center_board.reshape(1, 3),
                            rvec,
                            tvec,
                            K,
                            dist
                        )

                        u, v = image_center.reshape(2)

                        # ------------------
                        # SAVE POSE VALUES
                        # ------------------

                        pose = {
                            "x_cm": float(x / 10),
                            "y_cm": float(y / 10),
                            "z_cm": float(z / 10),

                            "distance_cm": float(
                                np.linalg.norm(center_cam) / 10
                            ),

                            "tx_mm": float(tvec[0, 0]),
                            "ty_mm": float(tvec[1, 0]),
                            "tz_mm": float(tvec[2, 0]),

                            "rvec_x": float(rvec[0, 0]),
                            "rvec_y": float(rvec[1, 0]),
                            "rvec_z": float(rvec[2, 0]),

                            "roll_deg": float(roll),
                            "pitch_deg": float(pitch),
                            "yaw_deg": float(yaw),

                            "center_u_px": float(u),
                            "center_v_px": float(v),

                            "reprojection_rmse_px": reproj_rmse
                        }

                        # ------------------
                        # DRAW OVERLAYS
                        # ------------------

                        cv2.drawChessboardCorners(
                            frame,
                            CHECKERBOARD,
                            corners,
                            True
                        )

                        cv2.drawFrameAxes(
                            frame,
                            K,
                            dist,
                            rvec,
                            tvec,
                            75
                        )

                        cv2.circle(
                            frame,
                            (int(u), int(v)),
                            6,
                            (0, 0, 255),
                            -1
                        )

                        labels = [
                            (
                                f"Distance: "
                                f"{pose['distance_cm']:.1f} cm | "
                                f"Depth Z: {pose['z_cm']:.1f} cm"
                            ),
                            (
                                f"X: {pose['x_cm']:.1f} "
                                f"Y: {pose['y_cm']:.1f} cm"
                            ),
                            (
                                f"Roll: {roll:.1f} "
                                f"Pitch: {pitch:.1f} "
                                f"Yaw: {yaw:.1f} deg"
                            ),
                            (
                                f"Reprojection RMSE: "
                                f"{reproj_rmse:.2f} px"
                            )
                        ]

                        for i, label in enumerate(labels):

                            cv2.putText(
                                frame,
                                label,
                                (15, 36 + 32 * i),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.64,
                                (0, 255, 0),
                                2
                            )

                if pose is None:

                    cv2.putText(
                        frame,
                        "No valid checkerboard pose",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 0, 255),
                        2
                    )

                # --------------------------
                # LOG AT APPROXIMATELY 1 HZ
                # --------------------------

                if elapsed >= next_log_time:

                    stamp = (
                        datetime.now()
                        .astimezone()
                        .isoformat(timespec="milliseconds")
                    )

                    row = {
                        "timestamp": stamp,
                        "elapsed_s": round(elapsed, 3),
                        "frame_index": frame_index,
                        "corners_detected": int(found),
                        "pose_valid": int(
                            pose is not None
                        )
                    }

                    if pose is not None:
                        row.update(pose)

                    writer.writerow(row)
                    csv_file.flush()

                    observation = {
                        "timestamp": stamp,
                        "elapsed_s": round(elapsed, 3),
                        "frame_index": frame_index,
                        "corners_detected": bool(found),
                        "corners_px": corner_pixels,
                        "pose": pose,
                        "rotation_matrix": rotation_matrix
                    }

                    json_file.write(
                        json.dumps(observation) + "\n"
                    )

                    json_file.flush()

                    if pose:
                        print(
                            f"{elapsed:5.1f}s "
                            f"Distance={pose['distance_cm']:.1f} cm "
                            f"Z={pose['z_cm']:.1f} cm"
                        )
                    else:
                        print(
                            f"{elapsed:5.1f}s | No pose"
                        )

                    next_log_time = int(elapsed) + 1

                # --------------------------
                # SAVE VIDEO
                # --------------------------

                cv2.putText(
                    frame,
                    f"REC {elapsed:.1f}s",
                    (15, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                video.write(frame)

                cv2.imshow(
                    "Checkerboard Tracking (Q = Stop)",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:

        cap.release()

        if video is not None:
            video.release()

        cv2.destroyAllWindows()


# ==========================================
# AUTOMATIC POST-PROCESSING
# ==========================================

def run_pipeline():

    scripts = [
        "checkerboard_motion_analysis.py",
        "checkerboard_3d_animation.py"
    ]

    for script in scripts:

        print(f"\nRunning {script} ...")

        result = subprocess.run([
            sys.executable,
            str(BASE_DIR / script),
            str(SESSION_DIR)
        ])

        if result.returncode != 0:
            print(
                f"WARNING: {script} failed. "
                "Recording data is still saved."
            )

    print("\nProcessing finished!")
    print("Results saved to:", SESSION_DIR)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    record()

    # Automatically runs AFTER recording stops
    run_pipeline()

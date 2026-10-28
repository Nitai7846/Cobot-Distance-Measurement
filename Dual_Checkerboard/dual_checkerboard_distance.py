#!/usr/bin/env python3
"""Dual checkerboard 6-DoF pose + three distances + video and 1-Hz logging.

Independent of the older MyCode project. Place webcam_calibration.npz next to
this script. Q stops and runs analysis + 3D animation. R resets A/B labels.

Identical boards cannot be uniquely identified after ambiguous crossings/occlusions.
Positions/distances refer to board CENTERS, not human/robot physical surfaces.
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

# -------- Configuration --------
CHECKERBOARD = (10, 7)             # Inner corners
SQUARE_SIZE_MM = 25.0
CAMERA_INDEX = 0
LOG_INTERVAL_S = 1.0

BASE_DIR = Path(__file__).resolve().parent
CALIB_PATH = BASE_DIR / "webcam_calibration.npz"
RESULTS_DIR = BASE_DIR / "Dual_Checkerboard_Results"

DETECTION_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

OBJECT_POINTS = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
OBJECT_POINTS[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    * SQUARE_SIZE_MM
)
CENTER_BOARD = np.array([
    (CHECKERBOARD[0] - 1) * SQUARE_SIZE_MM / 2,
    (CHECKERBOARD[1] - 1) * SQUARE_SIZE_MM / 2,
    0.0
], dtype=np.float64).reshape(3, 1)

COLOR_A = (255, 200, 50)   # BGR cyan
COLOR_B = (80, 200, 255)   # BGR yellow-orange
COLOR_PAIR = (90, 255, 120)

POSE_FIELDS = [
    "x_cm", "y_cm", "z_cm", "distance_cm",
    "tx_mm", "ty_mm", "tz_mm",
    "rvec_x", "rvec_y", "rvec_z",
    "roll_deg", "pitch_deg", "yaw_deg",
    "center_u_px", "center_v_px", "reprojection_rmse_px"
]
FIELDS = (
    ["timestamp", "elapsed_s", "frame_index", "tracking_epoch", "detected_count"]
    + [f"{label}_pose_valid" for label in ("a", "b")]
    + [f"{label}_{col}" for label in ("a", "b") for col in POSE_FIELDS]
    + ["pair_distance_cm"]
)


def draw_label(frame, text, x, y, color=(255, 255, 255), size=0.65):
    origin = (int(x), int(y))
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2, cv2.LINE_AA)


def detect_checkerboards(gray, max_boards=2):
    """Sequential full-frame detection; mask board #1 before looking for #2."""
    working = gray.copy()
    detections = []
    for _ in range(max_boards):
        found, corners = cv2.findChessboardCorners(
            working, CHECKERBOARD, DETECTION_FLAGS
        )
        if not found:
            break

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA
        )
        detections.append(corners)

        grid = corners.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
        dx = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
        dy = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()
        square_px = float(np.median(np.concatenate((dx, dy))))
        padding = int(np.clip(round(square_px), 5, 75))

        mask = np.zeros_like(gray)
        hull = cv2.convexHull(np.round(corners.reshape(-1, 2)).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
        )
        working[cv2.dilate(mask, kernel) != 0] = 127
    return detections


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


def estimate_board_pose(corners, K, dist):
    success, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if (not success or not np.isfinite(rvec).all()
            or not np.isfinite(tvec).all() or tvec[2, 0] <= 0):
        return None
    R, _ = cv2.Rodrigues(rvec)
    center_cm = ((R @ CENTER_BOARD + tvec) / 10.0).ravel()

    projected_center, _ = cv2.projectPoints(
        CENTER_BOARD.reshape(1, 3), rvec, tvec, K, dist
    )
    u, v = projected_center.reshape(2)
    reproj, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, K, dist)
    residual = reproj.reshape(-1, 2) - corners.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    roll, pitch, yaw = euler_zyx_degrees(R)

    return {
        "corners": corners,
        "rvec": rvec,
        "tvec": tvec,
        "center_cm": center_cm,
        "pixel_center": (int(round(u)), int(round(v))),
        "rmse_px": rmse,
        "roll_deg": float(roll),
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
    }


class BoardIdentityTracker:
    """Nearest-pixel tracking; labels are not guaranteed through occlusions."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized = False
        self.last_pixels = {"A": None, "B": None}

    def assign(self, detections):
        assigned = {"A": None, "B": None}
        if not self.initialized:
            if len(detections) < 2:
                return assigned
            ordered = sorted(detections, key=lambda d: d["pixel_center"][0])
            assigned["A"], assigned["B"] = ordered
            self.initialized = True
        elif len(detections) == 2:
            d0, d1 = detections
            pa, pb = [np.asarray(self.last_pixels[k], dtype=float) for k in ("A", "B")]
            p0, p1 = [np.asarray(d["pixel_center"], dtype=float) for d in (d0, d1)]
            same = np.linalg.norm(p0 - pa) + np.linalg.norm(p1 - pb)
            swapped = np.linalg.norm(p1 - pa) + np.linalg.norm(p0 - pb)
            assigned["A"], assigned["B"] = (d0, d1) if same <= swapped else (d1, d0)
        elif len(detections) == 1:
            p = np.asarray(detections[0]["pixel_center"], dtype=float)
            d_a = np.linalg.norm(p - self.last_pixels["A"])
            d_b = np.linalg.norm(p - self.last_pixels["B"])
            assigned["A" if d_a <= d_b else "B"] = detections[0]

        for label, pose in assigned.items():
            if pose is not None:
                self.last_pixels[label] = pose["pixel_center"]
        return assigned


def draw_board(frame, board, label, color, K, dist):
    if board is None:
        return
    cv2.drawChessboardCorners(frame, CHECKERBOARD, board["corners"], True)
    cv2.drawFrameAxes(frame, K, dist, board["rvec"], board["tvec"], 75.0)
    x, y = board["pixel_center"]
    cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 9, (0, 0, 0), 2, cv2.LINE_AA)
    draw_label(frame, label, x + 12, y - 12, color)


def pose_to_record(pose):
    if pose is None:
        return {}
    x, y, z = map(float, pose["center_cm"])
    tx, ty, tz = map(float, pose["tvec"].ravel())
    rx, ry, rz = map(float, pose["rvec"].ravel())
    u, v = pose["pixel_center"]
    return {
        "x_cm": x, "y_cm": y, "z_cm": z,
        "distance_cm": float(np.linalg.norm(pose["center_cm"])),
        "tx_mm": tx, "ty_mm": ty, "tz_mm": tz,
        "rvec_x": rx, "rvec_y": ry, "rvec_z": rz,
        "roll_deg": pose["roll_deg"],
        "pitch_deg": pose["pitch_deg"],
        "yaw_deg": pose["yaw_deg"],
        "center_u_px": u, "center_v_px": v,
        "reprojection_rmse_px": pose["rmse_px"],
    }


def record():
    if not CALIB_PATH.is_file():
        raise FileNotFoundError(f"Calibration missing: {CALIB_PATH}")

    with np.load(CALIB_PATH) as saved:
        K, dist = saved["K"], saved["dist"]
        calibrated_size = tuple(int(v) for v in saved["image_size"])
        rms = float(saved["rms"]) if "rms" in saved.files else None

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Could not access MacBook webcam")

    tracker = BoardIdentityTracker()
    video = None
    session = None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, calibrated_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calibrated_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Could not read camera")
        height, width = first.shape[:2]
        if (width, height) != calibrated_size:
            raise RuntimeError(
                f"Camera is {width}x{height}; calibration is {calibrated_size}"
            )
        nominal_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(nominal_fps) or nominal_fps <= 0:
            nominal_fps = 30.0

        session = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=False)
        video = cv2.VideoWriter(
            str(session / "recording.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), nominal_fps,
            (width, height)
        )
        if not video.isOpened():
            raise RuntimeError("Could not initialize MP4 writer")

        metadata = {
            "session": session.name,
            "checkerboard_inner_corners": list(CHECKERBOARD),
            "square_size_mm": SQUARE_SIZE_MM,
            "camera_matrix": K.tolist(),
            "distortion_coefficients": dist.ravel().tolist(),
            "calibration_rms_px": rms,
            "image_size": [width, height],
            "video_nominal_fps": nominal_fps,
            "logging_interval_s": LOG_INTERVAL_S,
            "coordinate_frame": "OpenCV camera X right, Y down, Z forward",
            "pose_frame": "board-to-camera; rvec/tvec origin = first inner corner",
            "distance_reference": "physical centers of printed checkerboards",
            "board_identity": "A initially leftmost; nearest-pixel matching thereafter; R resets labels",
        }
        (session / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print("Recording! Whole, non-overlapping boards anywhere in frame.")
        print("Q = stop and analyze; R = reset A/B labels.")
        print("Saving to:", session)
        print("Note: camera video uses nominal FPS; elapsed_s uses monotonic time.")
        if rms is not None and rms > 2.0:
            print(f"CAUTION: calibration RMS = {rms:.2f} px")

        frame_index = 0
        tracking_epoch = 0
        start = time.monotonic()
        next_log_time = 0.0
        with (
            (session / "pose_1hz.csv").open("w", newline="") as csv_file,
            (session / "observations_1hz.jsonl").open("w") as json_file,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame unavailable; stopping")
                    break
                frame_index += 1
                elapsed = time.monotonic() - start
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners_list = detect_checkerboards(gray, max_boards=2)
                found_poses = [estimate_board_pose(c, K, dist) for c in corners_list]
                found_poses = [p for p in found_poses if p is not None]
                assigned = tracker.assign(found_poses)
                a, b = assigned["A"], assigned["B"]

                draw_board(frame, a, "A", COLOR_A, K, dist)
                draw_board(frame, b, "B", COLOR_B, K, dist)

                pair_cm = None
                if a is not None and b is not None:
                    pair_cm = float(np.linalg.norm(a["center_cm"] - b["center_cm"]))
                    cv2.line(frame, a["pixel_center"], b["pixel_center"],
                             COLOR_PAIR, 2, cv2.LINE_AA)
                draw_label(frame, f"Boards: {len(found_poses)}/2", 15, 34)
                for label, board, y, color in (("A", a, 66, COLOR_A),
                                               ("B", b, 98, COLOR_B)):
                    text = (
                        f"Camera -> {label}: {np.linalg.norm(board['center_cm']):.1f} cm "
                        f"(RMSE {board['rmse_px']:.1f} px)"
                        if board is not None else f"Camera -> {label}: not detected"
                    )
                    draw_label(frame, text, 15, y, color)
                draw_label(frame, (
                    f"A <-> B: {pair_cm:.1f} cm" if pair_cm is not None
                    else "A <-> B: unavailable (need both boards)"
                ), 15, 130, COLOR_PAIR)
                if not tracker.initialized:
                    draw_label(frame, "Show BOTH boards to initialize A/B",
                               15, 162, size=0.55)
                draw_label(frame, f"REC {elapsed:.1f}s | Q = stop | R = reset A/B",
                           15, height - 18, size=0.55)

                if elapsed >= next_log_time:
                    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    row = {
                        "timestamp": stamp,
                        "elapsed_s": round(elapsed, 3),
                        "frame_index": frame_index,
                        "tracking_epoch": tracking_epoch,
                        "detected_count": len(found_poses),
                        "a_pose_valid": int(a is not None),
                        "b_pose_valid": int(b is not None),
                        "pair_distance_cm": pair_cm,
                    }
                    for prefix, pose in (("a", a), ("b", b)):
                        row.update({f"{prefix}_{key}": val
                                    for key, val in pose_to_record(pose).items()})
                    writer.writerow(row)
                    csv_file.flush()
                    json_file.write(json.dumps({
                        "timestamp": stamp,
                        "elapsed_s": round(elapsed, 3),
                        "frame_index": frame_index,
                        "tracking_epoch": tracking_epoch,
                        "boards": {
                            prefix: {
                                "corners_px": p["corners"].reshape(-1, 2).tolist(),
                                "rvec": p["rvec"].ravel().tolist(),
                                "tvec_mm": p["tvec"].ravel().tolist(),
                            } if p is not None else None
                            for prefix, p in (("a", a), ("b", b))
                        },
                    }) + "\n")
                    json_file.flush()
                    print(f"{elapsed:5.1f}s | A={row.get('a_distance_cm', '-') or '-'} "
                          f"B={row.get('b_distance_cm', '-') or '-'} "
                          f"A-B={pair_cm if pair_cm is not None else '-'}")
                    next_log_time = (int(elapsed / LOG_INTERVAL_S) + 1) * LOG_INTERVAL_S

                video.write(frame)
                cv2.imshow("Dual Checkerboard: Q to stop", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    tracker.reset()
                    tracking_epoch += 1
                    print("A/B labels reset (new tracking epoch).")
    finally:
        cap.release()
        if video is not None:
            video.release()
        cv2.destroyAllWindows()
    return session


def run_pipeline(session):
    for script in ("dual_checkerboard_motion_analysis.py",
                   "dual_checkerboard_3d_animation.py"):
        print(f"\nRunning {script} ...", flush=True)
        result = subprocess.run([sys.executable, str(BASE_DIR / script), str(session)])
        if result.returncode:
            print(f"WARNING: {script} failed. Raw video and CSV are still saved.")
    print("\nSession saved to:", session)


if __name__ == "__main__":
    session_dir = record()
    if session_dir is not None:
        run_pipeline(session_dir)

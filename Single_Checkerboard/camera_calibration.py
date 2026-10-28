#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 02:13:33 2026

@author: nitaishah
"""

import cv2
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

CHECKERBOARD = (10, 7)
SQUARE_SIZE = 25.0  # mm

SAVE_PATH = Path(__file__).resolve().parent / "webcam_calibration.npz"

# Prepare 3D checkerboard corner coordinates
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float32
)

objp[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2) * SQUARE_SIZE
)

object_points = []
image_points = []

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================================
# INITIALIZE CAMERA
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press C to capture checkerboard views")
print("Press ENTER to calibrate (minimum 15 views)")
print("Press Q to quit")

image_size = None

# ==========================================
# COLLECT CALIBRATION IMAGES
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            cv2.drawChessboardCorners(
                frame, CHECKERBOARD, corners, found
            )

        cv2.putText(
            frame,
            f"Samples: {len(image_points)}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("Camera Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and found:

            object_points.append(objp.copy())
            image_points.append(corners.copy())

            print(f"Captured sample {len(image_points)}")

        if key in (10, 13):

            if len(image_points) < 15:
                print("Capture at least 15 images")
                continue

            print("Calibrating camera...")

            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None
            )

            print("\nCalibration completed!")
            print("\nCamera Matrix:\n", K)
            print("\nDistortion Coefficients:\n", dist)
            print("\nRMS Reprojection Error:", rms)

            np.savez(
                SAVE_PATH,
                K=K,
                dist=dist,
                image_size=np.array(image_size),
                rms=rms
            )

            print(f"\nCalibration saved to: {SAVE_PATH}")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
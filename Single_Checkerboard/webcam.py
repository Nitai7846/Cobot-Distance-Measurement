#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 01:58:53 2026

@author: nitaishah
"""

import cv2
import os
from datetime import datetime

# 1. Define output directory
output_dir = "/Users/nitaishah/Desktop/Distance finding code/MyCode/Sample_Videos"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. Generate a unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"webcam_{timestamp}.mp4")

# 3. Initialize MacBook webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not access webcam")

# Request camera resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 4. Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    30.0,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not initialize video writer")

print(f"Recording started: {output_path}")
print("Press Q to stop recording.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Save frame
        out.write(frame)

        # Display live video
        cv2.imshow("MacBook Webcam - Recording", frame)

        # Stop recording when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # 5. Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Video saved successfully: {output_path}")
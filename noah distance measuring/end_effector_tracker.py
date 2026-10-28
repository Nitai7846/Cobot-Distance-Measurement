"""
end_effector_tracker.py

CPU-efficient colored-marker tracking for a robot end effector.

Default target color:
    Orange, BGR = (0, 165, 255)

Press Q to quit when running this file directly.
"""

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


class EndEffectorColorTracker:
    """Track a colored circular marker using HSV thresholding."""

    def __init__(
        self,
        target_bgr=(0, 165, 255), # COLOR
        hue_tolerance: int = 10,
        minimum_saturation: int = 120,
        minimum_value: int = 100,
        minimum_area: float = 80.0,
        maximum_area: float = 50000.0,
        minimum_circularity: float = 0.30,
        maximum_tracking_jump: float = 180.0,
        reset_after_missed_frames: int = 10,
    ):
        self.target_bgr = target_bgr
        self.hue_tolerance = hue_tolerance
        self.minimum_saturation = minimum_saturation
        self.minimum_value = minimum_value
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.minimum_circularity = minimum_circularity
        self.maximum_tracking_jump = maximum_tracking_jump
        self.reset_after_missed_frames = reset_after_missed_frames

        color_array = np.uint8([[target_bgr]])
        target_hsv = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        self.target_hue = int(target_hsv[0, 0, 0])

        self.previous_position: Optional[Point] = None
        self.missed_frames = 0

        self.kernel = np.ones((3, 3), dtype=np.uint8)

    def _create_mask(self, hsv_frame):
        """Create an HSV mask, including hue wraparound when needed."""
        low_hue = self.target_hue - self.hue_tolerance
        high_hue = self.target_hue + self.hue_tolerance

        if low_hue < 0:
            lower_1 = np.array(
                [0, self.minimum_saturation, self.minimum_value],
                dtype=np.uint8,
            )
            upper_1 = np.array(
                [high_hue, 255, 255],
                dtype=np.uint8,
            )

            lower_2 = np.array(
                [180 + low_hue, self.minimum_saturation, self.minimum_value],
                dtype=np.uint8,
            )
            upper_2 = np.array(
                [180, 255, 255],
                dtype=np.uint8,
            )

            mask = cv2.bitwise_or(
                cv2.inRange(hsv_frame, lower_1, upper_1),
                cv2.inRange(hsv_frame, lower_2, upper_2),
            )

        elif high_hue > 180:
            lower_1 = np.array(
                [low_hue, self.minimum_saturation, self.minimum_value],
                dtype=np.uint8,
            )
            upper_1 = np.array(
                [180, 255, 255],
                dtype=np.uint8,
            )

            lower_2 = np.array(
                [0, self.minimum_saturation, self.minimum_value],
                dtype=np.uint8,
            )
            upper_2 = np.array(
                [high_hue - 180, 255, 255],
                dtype=np.uint8,
            )

            mask = cv2.bitwise_or(
                cv2.inRange(hsv_frame, lower_1, upper_1),
                cv2.inRange(hsv_frame, lower_2, upper_2),
            )

        else:
            lower = np.array(
                [low_hue, self.minimum_saturation, self.minimum_value],
                dtype=np.uint8,
            )
            upper = np.array(
                [high_hue, 255, 255],
                dtype=np.uint8,
            )
            mask = cv2.inRange(hsv_frame, lower, upper)

        # Remove isolated noise, then fill small holes.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=2,
        )

        return mask

    def detect(self, frame, draw: bool = False) -> Optional[dict]:
        """
        Detect the marker.

        Returns:
            {
                "position": (x, y),
                "area": contour_area,
                "circularity": value,
                "confidence": value,
                "bbox": (x, y, width, height),
                "mask": mask
            }

        Returns None when no acceptable marker is found.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._create_mask(hsv_frame)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if not self.minimum_area <= area <= self.maximum_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.minimum_circularity:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            position = (float(center_x), float(center_y))

            if self.previous_position is not None:
                jump = math.dist(position, self.previous_position)

                if jump > self.maximum_tracking_jump:
                    continue
            else:
                jump = 0.0

            candidates.append(
                {
                    "position": position,
                    "area": float(area),
                    "circularity": float(circularity),
                    "contour": contour,
                    "jump": float(jump),
                    "bbox": cv2.boundingRect(contour),
                }
            )

        if not candidates:
            self.missed_frames += 1

            if self.missed_frames >= self.reset_after_missed_frames:
                self.previous_position = None

            return None

        if self.previous_position is None:
            # Initial acquisition: choose the largest acceptable marker.
            selected = max(candidates, key=lambda item: item["area"])
        else:
            # Continuing track: choose the marker closest to its last location.
            selected = min(candidates, key=lambda item: item["jump"])

        self.previous_position = selected["position"]
        self.missed_frames = 0

        area_confidence = min(
            1.0,
            selected["area"] / max(self.minimum_area * 5.0, 1.0),
        )
        shape_confidence = min(1.0, selected["circularity"])
        confidence = area_confidence * shape_confidence

        result = {
            "position": selected["position"],
            "area": selected["area"],
            "circularity": selected["circularity"],
            "confidence": confidence,
            "bbox": selected["bbox"],
            "mask": mask,
        }

        if draw:
            x, y, width, height = selected["bbox"]
            center = (
                int(round(selected["position"][0])),
                int(round(selected["position"][1])),
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            cv2.circle(frame, center, 6, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"End effector: ({center[0]}, {center[1]})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return result


class SimpleMotionEstimator:
    """Estimate smoothed velocity from a stream of 2D positions."""

    def __init__(
        self,
        position_alpha: float = 0.45,
        velocity_alpha: float = 0.35,
        maximum_gap_seconds: float = 0.5,
    ):
        self.position_alpha = position_alpha
        self.velocity_alpha = velocity_alpha
        self.maximum_gap_seconds = maximum_gap_seconds

        self.position: Optional[Point] = None
        self.velocity: Optional[Point] = None
        self.timestamp: Optional[float] = None

    def update(
        self,
        measurement: Optional[Point],
        timestamp: float,
    ):
        if measurement is None:
            return None, None

        measured_x, measured_y = measurement

        if (
            self.position is None
            or self.timestamp is None
            or timestamp - self.timestamp > self.maximum_gap_seconds
        ):
            self.position = (measured_x, measured_y)
            self.velocity = None
            self.timestamp = timestamp
            return self.position, self.velocity

        dt = timestamp - self.timestamp
        if dt <= 0:
            return self.position, self.velocity

        filtered_x = (
            self.position_alpha * measured_x
            + (1.0 - self.position_alpha) * self.position[0]
        )
        filtered_y = (
            self.position_alpha * measured_y
            + (1.0 - self.position_alpha) * self.position[1]
        )

        raw_velocity_x = (filtered_x - self.position[0]) / dt
        raw_velocity_y = (filtered_y - self.position[1]) / dt

        if self.velocity is None:
            filtered_velocity = (raw_velocity_x, raw_velocity_y)
        else:
            filtered_velocity = (
                self.velocity_alpha * raw_velocity_x
                + (1.0 - self.velocity_alpha) * self.velocity[0],
                self.velocity_alpha * raw_velocity_y
                + (1.0 - self.velocity_alpha) * self.velocity[1],
            )

        self.position = (filtered_x, filtered_y)
        self.velocity = filtered_velocity
        self.timestamp = timestamp

        return self.position, self.velocity


def main():
    camera_index = 0

    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = EndEffectorColorTracker(
        target_bgr=(0, 165, 255),
    )
    motion = SimpleMotionEstimator()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp = time.perf_counter()
            detection = tracker.detect(frame, draw=True)

            measured_position = (
                detection["position"] if detection is not None else None
            )

            position, velocity = motion.update(
                measured_position,
                timestamp,
            )

            if position is not None and velocity is not None:
                speed = math.hypot(velocity[0], velocity[1])

                cv2.putText(
                    frame,
                    (
                        f"Velocity: ({velocity[0]:.1f}, "
                        f"{velocity[1]:.1f}) px/s"
                    ),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Speed: {speed:.1f} px/s",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            if detection is None:
                cv2.putText(
                    frame,
                    "End-effector marker not detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("End-Effector Color Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
combined_tracker.py

Combines:
- MediaPipe operator pose tracking
- HSV end-effector color tracking
- Smoothed 2D velocity estimation
- CSV logging of human/end-effector relative velocity

Run:
    python combined_tracker.py

Press Q to stop. The CSV file is closed safely when the program exits.
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2

from pose_tracker import PoseTracker
from end_effector_tracker import (
    EndEffectorColorTracker,
    SimpleMotionEstimator,
)


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CSV_FILENAME = "relative_velocity.csv"

# 0.0 logs every processed frame.
# Set to 1.0 if you literally want one CSV update per second.
CSV_LOG_INTERVAL_SECONDS = 0.0

# Drawing adds some CPU work. Disable this when benchmarking performance.
DRAW_FULL_SKELETON = False
SHOW_MASK_WINDOW = False


COLORS = {'red': (0, 0, 255), 'orange': (0, 165, 255),
          'yellow': (255, 255, 0), 'lime': (57, 255, 20),
          'cyan': (0, 255, 255), 'test_green': (192, 248, 88)}
END_EFFECTOR_BGR = COLORS["test_green"]

TRACKED_HUMAN_POINTS = (
    "nose",
    "torso",
    "left_wrist",
    "right_wrist",
)


class CombinedMotionTracker:
    """Maintain independent motion filters for each tracked object."""

    def __init__(self):
        self.robot_motion = SimpleMotionEstimator(
            position_alpha=0.50,
            velocity_alpha=0.35,
            maximum_gap_seconds=0.5,
        )

        self.human_motion = {
            point_name: SimpleMotionEstimator(
                position_alpha=0.35,
                velocity_alpha=0.30,
                maximum_gap_seconds=0.5,
            )
            for point_name in TRACKED_HUMAN_POINTS
        }

    def update_robot(
        self,
        measurement: Optional[Point],
        timestamp: float,
    ):
        return self.robot_motion.update(measurement, timestamp)

    def update_human(
        self,
        point_name: str,
        measurement: Optional[Point],
        timestamp: float,
    ):
        return self.human_motion[point_name].update(
            measurement,
            timestamp,
        )


def calculate_relative_metrics(
    human_position: Optional[Point],
    human_velocity: Optional[Point],
    robot_position: Optional[Point],
    robot_velocity: Optional[Point],
) -> dict:
    """
    Calculate relative position and velocity.

    relative_velocity = human_velocity - robot_velocity

    separation_rate:
        Positive -> moving apart
        Negative -> moving together

    closing_speed:
        Positive -> moving together
        Negative -> moving apart
    """
    result = {
        "distance": None,
        "relative_position": None,
        "relative_velocity": None,
        "relative_speed": None,
        "separation_rate": None,
        "closing_speed": None,
    }

    if human_position is None or robot_position is None:
        return result

    relative_x = human_position[0] - robot_position[0]
    relative_y = human_position[1] - robot_position[1]
    distance = math.hypot(relative_x, relative_y)

    result["relative_position"] = (relative_x, relative_y)
    result["distance"] = distance

    if human_velocity is None or robot_velocity is None:
        return result

    relative_velocity_x = human_velocity[0] - robot_velocity[0]
    relative_velocity_y = human_velocity[1] - robot_velocity[1]

    result["relative_velocity"] = (
        relative_velocity_x,
        relative_velocity_y,
    )
    result["relative_speed"] = math.hypot(
        relative_velocity_x,
        relative_velocity_y,
    )

    if distance > 1e-6:
        unit_x = relative_x / distance
        unit_y = relative_y / distance

        separation_rate = (
            unit_x * relative_velocity_x
            + unit_y * relative_velocity_y
        )

        result["separation_rate"] = separation_rate
        result["closing_speed"] = -separation_rate

    return result


def value_or_blank(value):
    """Convert missing values to blank CSV fields."""
    return "" if value is None else value


def component_or_blank(vector, index):
    if vector is None:
        return ""
    return vector[index]


def draw_tracking_point(
    frame,
    point_name: str,
    position: Point,
    color,
):
    center = (
        int(round(position[0])),
        int(round(position[1])),
    )

    cv2.circle(frame, center, 6, color, -1)

    cv2.putText(
        frame,
        point_name,
        (center[0] + 7, center[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )


def main():
    output_path = Path(CSV_FILENAME).resolve()

    capture = cv2.VideoCapture(CAMERA_INDEX)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    pose_tracker = PoseTracker(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        landmark_visibility_threshold=0.5,
    )

    end_effector_tracker = EndEffectorColorTracker(
        target_bgr=END_EFFECTOR_BGR,
        hue_tolerance=10,
        minimum_saturation=120,
        minimum_value=100,
        minimum_area=80,
        minimum_circularity=0.30,
    )

    motion_tracker = CombinedMotionTracker()

    csv_fields = [
        "wall_time",
        "elapsed_seconds",
        "frame_number",
        "point_name",
        "valid_position",
        "valid_relative_velocity",
        "human_confidence",
        "robot_confidence",
        "human_x_px",
        "human_y_px",
        "robot_x_px",
        "robot_y_px",
        "human_vx_px_s",
        "human_vy_px_s",
        "robot_vx_px_s",
        "robot_vy_px_s",
        "relative_x_px",
        "relative_y_px",
        "distance_px",
        "relative_vx_px_s",
        "relative_vy_px_s",
        "relative_speed_px_s",
        "separation_rate_px_s",
        "closing_speed_px_s",
    ]

    point_colors = {
        "nose": (255, 255, 0),
        "torso": (255, 0, 255),
        "left_wrist": (0, 255, 255),
        "right_wrist": (0, 165, 255),
    }

    start_time = time.perf_counter()
    previous_frame_time = start_time
    last_csv_log_time = float("-inf")
    last_csv_flush_time = start_time
    displayed_fps = 0.0
    frame_number = 0

    print(f"Writing CSV data to: {output_path}")
    print("Press Q in the video window to stop.")

    try:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    print("Failed to read a camera frame.")
                    break

                timestamp = time.perf_counter()
                elapsed = timestamp - start_time
                frame_number += 1

                # -----------------------------------------------------------
                # Run both detectors on the same captured frame.
                # -----------------------------------------------------------

                pose_results = pose_tracker.process(
                    frame,
                    draw=DRAW_FULL_SKELETON,
                )

                robot_detection = end_effector_tracker.detect(
                    frame,
                    draw=True,
                )

                robot_measurement = (
                    robot_detection["position"]
                    if robot_detection is not None
                    else None
                )

                robot_position, robot_velocity = (
                    motion_tracker.update_robot(
                        robot_measurement,
                        timestamp,
                    )
                )

                robot_confidence = (
                    robot_detection["confidence"]
                    if robot_detection is not None
                    else None
                )

                should_log = (
                    CSV_LOG_INTERVAL_SECONDS <= 0.0
                    or timestamp - last_csv_log_time
                    >= CSV_LOG_INTERVAL_SECONDS
                )

                # -----------------------------------------------------------
                # Process each human point independently.
                # -----------------------------------------------------------

                for point_name in TRACKED_HUMAN_POINTS:
                    detection = pose_results[point_name]

                    human_measurement = (
                        detection["position"]
                        if detection is not None
                        else None
                    )

                    human_confidence = (
                        detection["confidence"]
                        if detection is not None
                        else None
                    )

                    human_position, human_velocity = (
                        motion_tracker.update_human(
                            point_name,
                            human_measurement,
                            timestamp,
                        )
                    )

                    metrics = calculate_relative_metrics(
                        human_position,
                        human_velocity,
                        robot_position,
                        robot_velocity,
                    )

                    if human_measurement is not None:
                        draw_tracking_point(
                            frame,
                            point_name,
                            human_measurement,
                            point_colors[point_name],
                        )

                    # Draw line from each visible human point to the robot.
                    if (
                        human_position is not None
                        and robot_position is not None
                    ):
                        human_center = (
                            int(round(human_position[0])),
                            int(round(human_position[1])),
                        )
                        robot_center = (
                            int(round(robot_position[0])),
                            int(round(robot_position[1])),
                        )

                        cv2.line(
                            frame,
                            human_center,
                            robot_center,
                            point_colors[point_name],
                            1,
                        )

                    # CSV uses long format: one row per human point.
                    if should_log:
                        valid_position = (
                            human_position is not None
                            and robot_position is not None
                        )
                        valid_velocity = (
                            metrics["relative_velocity"] is not None
                        )

                        writer.writerow(
                            {
                                "wall_time": datetime.now().isoformat(
                                    timespec="milliseconds"
                                ),
                                "elapsed_seconds": f"{elapsed:.6f}",
                                "frame_number": frame_number,
                                "point_name": point_name,
                                "valid_position": int(valid_position),
                                "valid_relative_velocity": int(
                                    valid_velocity
                                ),
                                "human_confidence": value_or_blank(
                                    human_confidence
                                ),
                                "robot_confidence": value_or_blank(
                                    robot_confidence
                                ),
                                "human_x_px": component_or_blank(
                                    human_position, 0
                                ),
                                "human_y_px": component_or_blank(
                                    human_position, 1
                                ),
                                "robot_x_px": component_or_blank(
                                    robot_position, 0
                                ),
                                "robot_y_px": component_or_blank(
                                    robot_position, 1
                                ),
                                "human_vx_px_s": component_or_blank(
                                    human_velocity, 0
                                ),
                                "human_vy_px_s": component_or_blank(
                                    human_velocity, 1
                                ),
                                "robot_vx_px_s": component_or_blank(
                                    robot_velocity, 0
                                ),
                                "robot_vy_px_s": component_or_blank(
                                    robot_velocity, 1
                                ),
                                "relative_x_px": component_or_blank(
                                    metrics["relative_position"], 0
                                ),
                                "relative_y_px": component_or_blank(
                                    metrics["relative_position"], 1
                                ),
                                "distance_px": value_or_blank(
                                    metrics["distance"]
                                ),
                                "relative_vx_px_s": component_or_blank(
                                    metrics["relative_velocity"], 0
                                ),
                                "relative_vy_px_s": component_or_blank(
                                    metrics["relative_velocity"], 1
                                ),
                                "relative_speed_px_s": value_or_blank(
                                    metrics["relative_speed"]
                                ),
                                "separation_rate_px_s": value_or_blank(
                                    metrics["separation_rate"]
                                ),
                                "closing_speed_px_s": value_or_blank(
                                    metrics["closing_speed"]
                                ),
                            }
                        )

                    # Display the torso separation rate because it is the
                    # strongest indicator of whole-body withdrawal.
                    if (
                        point_name == "torso"
                        and metrics["separation_rate"] is not None
                    ):
                        cv2.putText(
                            frame,
                            (
                                "Torso separation rate: "
                                f"{metrics['separation_rate']:.1f} px/s"
                            ),
                            (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                        )

                        cv2.putText(
                            frame,
                            (
                                "Torso closing speed: "
                                f"{metrics['closing_speed']:.1f} px/s"
                            ),
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                        )

                if should_log:
                    last_csv_log_time = timestamp

                # Flush approximately once per second so data survives most
                # unexpected program exits without forcing a write every row.
                if timestamp - last_csv_flush_time >= 1.0:
                    csv_file.flush()
                    last_csv_flush_time = timestamp

                # -----------------------------------------------------------
                # Frame-rate display
                # -----------------------------------------------------------

                frame_dt = timestamp - previous_frame_time
                previous_frame_time = timestamp

                if frame_dt > 0:
                    instantaneous_fps = 1.0 / frame_dt
                    displayed_fps = (
                        0.9 * displayed_fps
                        + 0.1 * instantaneous_fps
                        if displayed_fps > 0
                        else instantaneous_fps
                    )

                cv2.putText(
                    frame,
                    f"FPS: {displayed_fps:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if robot_detection is None:
                    cv2.putText(
                        frame,
                        "End-effector marker lost",
                        (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Combined Human-Robot Tracking", frame)

                if SHOW_MASK_WINDOW and robot_detection is not None:
                    cv2.imshow(
                        "End-Effector Mask",
                        robot_detection["mask"],
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nTracking stopped by user.")

    finally:
        pose_tracker.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
"""
pose_tracker.py

Lightweight CPU human-pose tracking using MediaPipe Pose Landmarker (Tasks API).

Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' -OutFile 'pose_landmarker_lite.task'

Tracked operator points:
- nose
- torso (average of shoulders and hips)
- left wrist
- right wrist

Press Q to quit when running this file directly.
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp

Point = Tuple[float, float]


class PoseTracker:
    """CPU-oriented MediaPipe pose tracker (Tasks API)."""

    # Landmark indices for MediaPipe Pose
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    def __init__(
        self,
        model_path: str = "pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_visibility_threshold: float = 0.5,
    ):
        self.visibility_threshold = landmark_visibility_threshold

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _get_landmark(
        self,
        landmarks,
        landmark_id: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        landmark = landmarks[landmark_id]
        visibility = getattr(landmark, "visibility", 1.0)

        if visibility < self.visibility_threshold:
            return None

        return {
            "position": (
                float(landmark.x * frame_width),
                float(landmark.y * frame_height),
            ),
            "confidence": float(visibility),
        }

    def process(self, frame, draw: bool = False) -> Dict[str, Optional[dict]]:
        frame_height, frame_width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        output = {
            "nose": None,
            "torso": None,
            "left_wrist": None,
            "right_wrist": None,
        }

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return output

        landmarks = results.pose_landmarks[0]

        output["nose"] = self._get_landmark(
            landmarks,
            self.NOSE_IDX,
            frame_width,
            frame_height,
        )

        output["left_wrist"] = self._get_landmark(
            landmarks,
            self.LEFT_WRIST_IDX,
            frame_width,
            frame_height,
        )

        output["right_wrist"] = self._get_landmark(
            landmarks,
            self.RIGHT_WRIST_IDX,
            frame_width,
            frame_height,
        )

        # Create a stable torso point from shoulders and hips.
        torso_landmark_ids = [
            self.LEFT_SHOULDER_IDX,
            self.RIGHT_SHOULDER_IDX,
            self.LEFT_HIP_IDX,
            self.RIGHT_HIP_IDX,
        ]

        torso_points = []
        torso_confidences = []

        for landmark_id in torso_landmark_ids:
            result = self._get_landmark(
                landmarks,
                landmark_id,
                frame_width,
                frame_height,
            )

            if result is not None:
                torso_points.append(result["position"])
                torso_confidences.append(result["confidence"])

        if len(torso_points) >= 3:
            torso_x = sum(point[0] for point in torso_points) / len(torso_points)
            torso_y = sum(point[1] for point in torso_points) / len(torso_points)

            output["torso"] = {
                "position": (torso_x, torso_y),
                "confidence": min(torso_confidences),
            }

        if draw:
            colors = {
                "nose": (255, 255, 0),
                "torso": (255, 0, 255),
                "left_wrist": (0, 255, 255),
                "right_wrist": (0, 165, 255),
            }

            for name, detection in output.items():
                if detection is None:
                    continue

                x, y = detection["position"]
                point = (int(round(x)), int(round(y)))

                cv2.circle(frame, point, 6, colors[name], -1)
                cv2.putText(
                    frame,
                    name,
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[name],
                    2,
                )

        return output

    def close(self):
        self.landmarker.close()


def main():
    camera_index = 0
    frame_width = 640
    frame_height = 480

    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = PoseTracker()

    previous_time = time.perf_counter()
    displayed_fps = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            points = tracker.process(frame, draw=True)

            current_time = time.perf_counter()
            dt = current_time - previous_time
            previous_time = current_time

            if dt > 0:
                instantaneous_fps = 1.0 / dt
                displayed_fps = (
                    0.9 * displayed_fps + 0.1 * instantaneous_fps
                    if displayed_fps > 0
                    else instantaneous_fps
                )

            detected_names = [
                name for name, detection in points.items()
                if detection is not None
            ]

            cv2.putText(
                frame,
                f"FPS: {displayed_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Detected: " + ", ".join(detected_names),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("Operator Pose Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
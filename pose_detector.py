"""Local webcam capture and MediaPipe Pose Landmarker inference."""

from pathlib import Path
from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from config import CAMERA_FRAME_HEIGHT, CAMERA_FRAME_WIDTH, MODEL_PATH


@dataclass(frozen=True)
class PoseDetection:
    """Image-space and world-space landmarks for one detected pose."""

    image_landmarks: Sequence[object]
    world_landmarks: Sequence[object]
    image_size: tuple[int, int]


class PoseDetector:
    """Run MediaPipe Tasks pose inference on photos and camera frames."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        """Load the local Pose Landmarker model.

        Args:
            model_path: Path to a compatible local MediaPipe .task model.

        Raises:
            FileNotFoundError: If the local model asset is missing.
            RuntimeError: If MediaPipe cannot initialize the task.
        """
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. Place the .task model there."
            )
        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
        except Exception as error:
            raise RuntimeError(f"Could not load MediaPipe pose model: {error}") from error
        self._lock = Lock()

    def detect(self, bgr_image: np.ndarray) -> Sequence[object] | None:
        """Detect one person and return their world landmarks.

        Args:
            bgr_image: A non-empty OpenCV image in BGR color order.

        Returns:
            The first pose's 3D world landmarks, or None when no pose is found.

        Raises:
            ValueError: If the provided image is empty or malformed.
        """
        detection = self.detect_pose(bgr_image)
        return detection.world_landmarks if detection is not None else None

    def detect_pose(self, bgr_image: np.ndarray) -> PoseDetection | None:
        """Detect one pose and return image-space and world landmarks.

        Args:
            bgr_image: A non-empty OpenCV image in BGR color order.

        Returns:
            Both landmark sets for the first pose, or None when no pose is found.

        Raises:
            ValueError: If the provided image is empty or malformed.
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Cannot run pose detection on an empty image.")
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image.")

        frame_height, frame_width = bgr_image.shape[:2]
        scale = min(
            1.0,
            CAMERA_FRAME_WIDTH / frame_width,
            CAMERA_FRAME_HEIGHT / frame_height,
        )
        if scale < 1.0:
            bgr_image = cv2.resize(
                bgr_image,
                (int(frame_width * scale), int(frame_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frame_height, frame_width = bgr_image.shape[:2]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))
        with self._lock:
            result = self._landmarker.detect(image)
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        return PoseDetection(
            image_landmarks=result.pose_landmarks[0],
            world_landmarks=result.pose_world_landmarks[0],
            image_size=(frame_width, frame_height),
        )

    def close(self) -> None:
        """Release MediaPipe task resources.

        Returns:
            None.
        """
        with self._lock:
            self._landmarker.close()


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Open the requested camera or raise a clear error.

    Args:
        camera_index: OpenCV camera device index.

    Returns:
        An opened OpenCV VideoCapture instance.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera {camera_index} could not be opened.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def probe_camera(camera_index: int = 0) -> tuple[bool, str]:
    """Check that a camera opens and returns at least one frame.

    Args:
        camera_index: OpenCV camera device index.

    Returns:
        A success flag and a human-readable diagnostic message.
    """
    capture: cv2.VideoCapture | None = None
    try:
        capture = open_camera(camera_index)
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            return False, f"Camera {camera_index} opened but returned no frame."
        return True, f"Camera {camera_index} is available ({frame.shape[1]}x{frame.shape[0]})."
    except (RuntimeError, cv2.error) as error:
        return False, str(error)
    finally:
        if capture is not None:
            capture.release()

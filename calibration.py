"""Live posture baseline construction and local JSON persistence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config import CALIBRATION_PATH
from posture_analyzer import PostureAngles


@dataclass(frozen=True)
class Calibration:
    """Reference neck, forward-head, and normalized head-height metrics."""

    neck_tilt_degrees: float
    shoulder_line_tilt_degrees: float
    source: str
    forward_head_ratio: float = 0.0
    normalized_head_height: float = 0.0

    @property
    def angles(self) -> PostureAngles:
        """Return the baseline as a posture analyzer value object."""
        return PostureAngles(
            self.neck_tilt_degrees,
            self.shoulder_line_tilt_degrees,
            self.forward_head_ratio,
            self.normalized_head_height,
        )


def save_calibration(
    calibration: Calibration, path: Path = CALIBRATION_PATH
) -> None:
    """Save calibration data as UTF-8 JSON on this computer.

    Args:
        calibration: Baseline values to persist.
        path: Destination JSON file.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration | None:
    """Load and validate the saved baseline.

    Args:
        path: JSON file to load.

    Returns:
        The saved calibration, or None when no valid file exists.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        calibration = Calibration(
            neck_tilt_degrees=float(data["neck_tilt_degrees"]),
            shoulder_line_tilt_degrees=float(data["shoulder_line_tilt_degrees"]),
            source=str(data.get("source", data.get("photo_name", "camera"))),
            forward_head_ratio=float(data["forward_head_ratio"]),
            normalized_head_height=float(data["normalized_head_height"]),
        )
        if not all(
            0.0 <= angle <= 180.0
            for angle in (
                calibration.neck_tilt_degrees,
                calibration.shoulder_line_tilt_degrees,
            )
        ):
            return None
        return calibration
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def calibration_from_angles(angles: PostureAngles | None) -> Calibration:
    """Build a reference baseline from already validated pose angles.

    Args:
        angles: Neck and shoulder-line angles calculated from a live pose.

    Returns:
        The new baseline ready to be persisted.

    Raises:
        ValueError: If no usable upper-body pose was detected.
    """
    if angles is None:
        raise ValueError("Не удалось распознать позу, попробуй ещё раз")
    return Calibration(
        neck_tilt_degrees=angles.neck_tilt_degrees,
        shoulder_line_tilt_degrees=angles.shoulder_line_tilt_degrees,
        source="camera",
        forward_head_ratio=angles.forward_head_ratio,
        normalized_head_height=angles.normalized_head_height,
    )

"""Application settings and local data paths for SpineUp."""

from pathlib import Path
import os
import sys


APP_NAME = "SpineUp"
__version__ = "0.1.0"
PROJECT_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = Path(
    os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
) / APP_NAME


def resource_path(relative_path: str | Path) -> Path:
	"""Return a resource path for source runs and PyInstaller builds."""
	resource_dir = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))
	return resource_dir / relative_path


ICON_PNG_PATH = resource_path("logo/logo.png")
ICON_ICO_PATH = resource_path("logo/logo.ico")
ASSET_DIR = resource_path("assets")
MODEL_PATH = resource_path("models/pose_landmarker_lite.task")
DATA_DIR = Path.home() / ".spineup"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
DATABASE_PATH = DATA_DIR / "spineup.sqlite3"
SESSION_DATABASE_PATH = APP_DATA_DIR / "spineup.db"

CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 1.0
CAMERA_FRAME_WIDTH = 480
CAMERA_FRAME_HEIGHT = 360
PREVIEW_ACTIVE_INTERVAL_MS = 66
PREVIEW_BACKGROUND_INTERVAL_MS = 250
MAX_UNOBSERVED_SECONDS = 5.0
MIN_BAD_POSE_SECONDS = 3.0
SOFT_ALERT_SECONDS = 15.0
NOTIFICATION_SECONDS = 30.0
SITTING_BREAK_SECONDS = 60 * 60
BREAK_SNOOZE_SECONDS = 10 * 60
BREAK_DURATION_SECONDS = 60
POSTURE_DEVIATION_THRESHOLD_DEGREES = 15.0
MIN_LANDMARK_VISIBILITY = 0.65
METRIC_SMOOTHING_WINDOW = 3
MAX_NECK_ANGLE_CHANGE_PER_SECOND = 35.0
MAX_FORWARD_HEAD_CHANGE_PERCENT_PER_SECOND = 50.0
MAX_HEAD_HEIGHT_CHANGE_PER_SECOND = 0.8
CPU_METER_UPDATE_SECONDS = 1.0

# Positive neck-angle delta from the calibrated baseline that flags poor posture.
NECK_ANGLE_DEVIATION_THRESHOLD_DEGREES = 12.0
# Positive forward-head delta, in percentage points of shoulder width.
FORWARD_HEAD_DEVIATION_THRESHOLD_PERCENT = 8.0
# Normalized vertical head-height delta that flags a lowered head; tune from debug data.
NORMALIZED_HEAD_HEIGHT_DEVIATION_THRESHOLD = 0.06
GRACE_PERIOD_SECONDS = 3

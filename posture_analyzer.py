"""Calculate posture angles from MediaPipe world landmarks."""

from dataclasses import dataclass
from collections import deque
from math import atan2, degrees, hypot
from statistics import fmean
from time import monotonic
from typing import Protocol, Sequence

from config import (
    FORWARD_HEAD_DEVIATION_THRESHOLD_PERCENT,
    MAX_FORWARD_HEAD_CHANGE_PERCENT_PER_SECOND,
    MAX_HEAD_HEIGHT_CHANGE_PER_SECOND,
    MAX_NECK_ANGLE_CHANGE_PER_SECOND,
    MIN_LANDMARK_VISIBILITY,
    METRIC_SMOOTHING_WINDOW,
    NECK_ANGLE_DEVIATION_THRESHOLD_DEGREES,
    NORMALIZED_HEAD_HEIGHT_DEVIATION_THRESHOLD,
)


class Landmark3D(Protocol):
    """The coordinate fields required from a MediaPipe pose landmark."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PostureAngles:
    """Measured neck, shoulder, and normalized head-position metrics."""

    neck_tilt_degrees: float
    shoulder_line_tilt_degrees: float
    forward_head_ratio: float = 0.0
    normalized_head_height: float = 0.0


class PostureMetricFilter:
    """Smooth valid posture samples and reject isolated physically implausible jumps.

    A single sample that exceeds any configured metric rate is ignored. If the
    following sample confirms a similarly-sized, stable change, it is accepted
    as a real new pose rather than being rejected forever.
    """

    def __init__(
        self,
        window_size: int = METRIC_SMOOTHING_WINDOW,
        max_neck_change_per_second: float = MAX_NECK_ANGLE_CHANGE_PER_SECOND,
        max_forward_head_change_per_second: float = MAX_FORWARD_HEAD_CHANGE_PERCENT_PER_SECOND,
        max_head_height_change_per_second: float = MAX_HEAD_HEIGHT_CHANGE_PER_SECOND,
    ) -> None:
        """Initialize a bounded moving-average filter.

        Args:
            window_size: Number of accepted posture samples to average.
            max_neck_change_per_second: Maximum plausible neck-angle velocity.
            max_forward_head_change_per_second: Maximum plausible forward-head
                velocity in percentage points per second.
            max_head_height_change_per_second: Maximum plausible normalized
                head-height velocity per second.

        Returns:
            None.
        """
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.max_neck_change_per_second = max_neck_change_per_second
        self.max_forward_head_change_per_second = max_forward_head_change_per_second
        self.max_head_height_change_per_second = max_head_height_change_per_second
        self._samples: deque[PostureAngles] = deque(maxlen=window_size)
        self._last_sample_at: float | None = None
        self._pending_outlier: PostureAngles | None = None
        self._pending_outlier_at: float | None = None

    def reset(self, initial: PostureAngles | None = None, observed_at: float | None = None) -> None:
        """Clear the filter and optionally seed it with a baseline sample.

        Args:
            initial: Optional initial value to seed the moving average.
            observed_at: Monotonic timestamp for the initial sample.

        Returns:
            None.
        """
        self._samples.clear()
        self._last_sample_at = None
        self._pending_outlier = None
        self._pending_outlier_at = None
        if initial is not None:
            self._samples.append(initial)
            self._last_sample_at = observed_at if observed_at is not None else monotonic()

    def add(
        self, metrics: PostureAngles, observed_at: float | None = None
    ) -> PostureAngles | None:
        """Accept and smooth a sample, or return None when it is an isolated outlier.

        Args:
            metrics: Raw metrics from one confidence-validated camera frame.
            observed_at: Monotonic sample timestamp; current time is the default.

        Returns:
            Moving average after accepting the sample, or None when skipped.
        """
        sample_time = observed_at if observed_at is not None else monotonic()
        previous = self.value()
        if previous is not None and self._last_sample_at is not None:
            elapsed = max(sample_time - self._last_sample_at, 0.05)
            if self._is_outlier(metrics, previous, elapsed):
                if (
                    self._pending_outlier is None
                    or self._pending_outlier_at is None
                    or self._is_outlier(
                        metrics,
                        self._pending_outlier,
                        max(sample_time - self._pending_outlier_at, 0.05),
                    )
                ):
                    self._pending_outlier = metrics
                    self._pending_outlier_at = sample_time
                    return None

        self._samples.append(metrics)
        self._last_sample_at = sample_time
        self._pending_outlier = None
        self._pending_outlier_at = None
        return self.value()

    def value(self) -> PostureAngles | None:
        """Return the current moving average without adding a new sample.

        Returns:
            Average posture metrics, or None before the first accepted sample.
        """
        if not self._samples:
            return None
        return PostureAngles(
            neck_tilt_degrees=fmean(sample.neck_tilt_degrees for sample in self._samples),
            shoulder_line_tilt_degrees=fmean(
                sample.shoulder_line_tilt_degrees for sample in self._samples
            ),
            forward_head_ratio=fmean(
                sample.forward_head_ratio for sample in self._samples
            ),
            normalized_head_height=fmean(
                sample.normalized_head_height for sample in self._samples
            ),
        )

    def _is_outlier(
        self, candidate: PostureAngles, reference: PostureAngles, elapsed: float
    ) -> bool:
        """Compare each metric's velocity with its configured physical limit.

        Args:
            candidate: New raw sample.
            reference: Last smoothed sample or pending candidate.
            elapsed: Seconds between these samples.

        Returns:
            True if any metric changed faster than its permitted rate.
        """
        return (
            abs(candidate.neck_tilt_degrees - reference.neck_tilt_degrees)
            > self.max_neck_change_per_second * elapsed
            or abs(candidate.forward_head_ratio - reference.forward_head_ratio) * 100.0
            > self.max_forward_head_change_per_second * elapsed
            or abs(
                candidate.normalized_head_height - reference.normalized_head_height
            )
            > self.max_head_height_change_per_second * elapsed
        )


def _midpoint(first: Landmark3D, second: Landmark3D) -> tuple[float, float, float]:
    """Return the midpoint of two landmarks.

    Args:
        first: The first 3D landmark.
        second: The second 3D landmark.

    Returns:
        The midpoint as an (x, y, z) tuple.
    """
    return (
        (first.x + second.x) / 2.0,
        (first.y + second.y) / 2.0,
        (first.z + second.z) / 2.0,
    )


def _inclination_from_vertical(
    upper: tuple[float, float, float], lower: tuple[float, float, float]
) -> float:
    """Measure the 3D segment's inclination from the vertical axis.

    Args:
        upper: The upper endpoint as an (x, y, z) tuple.
        lower: The lower endpoint as an (x, y, z) tuple.

    Returns:
        An unsigned angle from 0 to 90 degrees.
    """
    horizontal_distance = hypot(upper[0] - lower[0], upper[2] - lower[2])
    vertical_distance = abs(upper[1] - lower[1])
    return degrees(atan2(horizontal_distance, vertical_distance))


def _shoulder_line_tilt(first: Landmark3D, second: Landmark3D) -> float:
    """Measure the unsigned shoulder-line angle relative to horizontal.

    Args:
        first: One shoulder world landmark.
        second: The opposite shoulder world landmark.

    Returns:
        The shoulder-line tilt from 0 to 90 degrees.
    """
    horizontal_distance = hypot(first.x - second.x, first.z - second.z)
    vertical_distance = abs(first.y - second.y)
    return degrees(atan2(vertical_distance, horizontal_distance))


def analyze_posture(
    landmarks: Sequence[Landmark3D],
    image_landmarks: Sequence[Landmark3D] | None = None,
    image_size: tuple[int, int] | None = None,
) -> PostureAngles | None:
    """Calculate neck, forward-head, and scale-normalized vertical head metrics.

    Args:
        landmarks: MediaPipe world landmarks in the documented 33-point order.
        image_landmarks: Matching normalized image landmarks for screen-horizontal
            head offset and pixel-coordinate head height. When omitted,
            image-derived metrics default to zero for compatibility.
        image_size: Input image dimensions as (width, height) pixels. Required when
            image_landmarks are supplied.

    Returns:
        The posture metrics, or None when required landmarks are unreliable.
    """
    if len(landmarks) < 13:
        return None

    required_indices = (7, 8, 11, 12)
    if any(
        getattr(landmarks[index], "visibility", 1.0) < MIN_LANDMARK_VISIBILITY
        or getattr(landmarks[index], "presence", 1.0) < MIN_LANDMARK_VISIBILITY
        for index in required_indices
    ):
        return None
    if image_landmarks is not None and (
        len(image_landmarks) < 13
        or any(
            getattr(image_landmarks[index], "visibility", 1.0)
            < MIN_LANDMARK_VISIBILITY
            or getattr(image_landmarks[index], "presence", 1.0)
            < MIN_LANDMARK_VISIBILITY
            for index in required_indices
        )
    ):
        return None

    ears = _midpoint(landmarks[7], landmarks[8])
    shoulders = _midpoint(landmarks[11], landmarks[12])
    forward_head_ratio = 0.0
    normalized_head_height = 0.0
    if image_landmarks is not None and len(image_landmarks) >= 13:
        if image_size is None:
            return None
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            return None
        image_ears = _midpoint(image_landmarks[7], image_landmarks[8])
        image_shoulders = _midpoint(image_landmarks[11], image_landmarks[12])
        shoulder_width_pixels = hypot(
            (image_landmarks[11].x - image_landmarks[12].x) * image_width,
            (image_landmarks[11].y - image_landmarks[12].y) * image_height,
        )
        if shoulder_width_pixels <= 0.0:
            return None
        forward_head_ratio = (
            (image_ears[0] - image_shoulders[0]) * image_width
        ) / shoulder_width_pixels
        normalized_head_height = (
            image_ears[1] * image_height
        ) / shoulder_width_pixels
    return PostureAngles(
        neck_tilt_degrees=_inclination_from_vertical(ears, shoulders),
        shoulder_line_tilt_degrees=_shoulder_line_tilt(
            landmarks[11], landmarks[12]
        ),
        forward_head_ratio=forward_head_ratio,
        normalized_head_height=normalized_head_height,
    )


def posture_deviation(
    current: PostureAngles, baseline: PostureAngles
) -> tuple[float, float, float]:
    """Compare measured angles with the user's calibrated baseline.

    Args:
        current: The current measured posture angles.
        baseline: The saved reference posture angles.

    Returns:
        Neck deviation, torso deviation, and the larger absolute deviation.
    """
    neck_deviation = current.neck_tilt_degrees - baseline.neck_tilt_degrees
    shoulder_deviation = (
        current.shoulder_line_tilt_degrees - baseline.shoulder_line_tilt_degrees
    )
    return neck_deviation, shoulder_deviation, max(
        abs(neck_deviation), abs(shoulder_deviation)
    )


def posture_metric_deviations(
    current: PostureAngles, baseline: PostureAngles
) -> tuple[float, float, float]:
    """Return neck, forward-head, and normalized vertical-position deviations.

    Args:
        current: Current pose metrics.
        baseline: Metrics saved at calibration time.

    Returns:
        Neck-angle delta in degrees, forward-head delta in percentage points of
        shoulder width, and normalized head-height delta.
    """
    neck_delta = current.neck_tilt_degrees - baseline.neck_tilt_degrees
    forward_head_delta_percent = (
        current.forward_head_ratio - baseline.forward_head_ratio
    ) * 100.0
    head_height_delta = (
        current.normalized_head_height - baseline.normalized_head_height
    )
    return neck_delta, forward_head_delta_percent, head_height_delta


def posture_exceeds_threshold(
    neck_delta_degrees: float,
    forward_head_delta_percent: float,
    normalized_head_height_delta: float = 0.0,
    neck_threshold_degrees: float = NECK_ANGLE_DEVIATION_THRESHOLD_DEGREES,
    forward_head_threshold_percent: float = FORWARD_HEAD_DEVIATION_THRESHOLD_PERCENT,
    normalized_head_height_threshold: float = NORMALIZED_HEAD_HEIGHT_DEVIATION_THRESHOLD,
) -> bool:
    """Return whether any metric exceeds its threshold toward poor posture.

    Args:
        neck_delta_degrees: Current-minus-baseline neck-angle delta; positive means
            the measured neck inclination increased.
        forward_head_delta_percent: Current-minus-baseline signed forward-head
            delta in shoulder-width percentage points.
        normalized_head_height_delta: Current-minus-baseline head-height ratio;
            positive means the head moved lower in the frame.
        neck_threshold_degrees: Absolute neck-angle tolerance in degrees.
        forward_head_threshold_percent: Absolute forward-head tolerance in percent.

    Returns:
        True only when a positive delta exceeds its metric-specific threshold.
    """
    return (
        neck_delta_degrees > neck_threshold_degrees
        or forward_head_delta_percent > forward_head_threshold_percent
        or normalized_head_height_delta > normalized_head_height_threshold
    )

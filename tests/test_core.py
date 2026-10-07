"""Unit tests for posture geometry, alert timing, and local history."""

import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from alert_manager import AlertManager
from calibration import calibration_from_angles, load_calibration, save_calibration
from database import PostureDatabase
from posture_analyzer import (
    PostureAngles,
    PostureMetricFilter,
    analyze_posture,
    posture_deviation,
    posture_exceeds_threshold,
    posture_metric_deviations,
)
from ui import AnalysisResult, SKELETON_CONNECTIONS, SpineUpWindow, draw_pose_skeleton
import numpy as np


@dataclass
class Landmark:
    """Minimal landmark value used by geometry tests."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


class PostureAnalyzerTests(unittest.TestCase):
    """Check baseline geometry and deviation calculations."""

    def test_neutral_pose_has_zero_inclination(self) -> None:
        """Return zero angles for vertically aligned landmark pairs."""
        landmarks = [Landmark(0.0, 0.0) for _ in range(33)]
        for index, y_value in ((7, 0.0), (8, 0.0), (11, 1.0), (12, 1.0), (23, 2.0), (24, 2.0)):
            landmarks[index] = Landmark(0.0, y_value)
        result = analyze_posture(landmarks)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.neck_tilt_degrees, 0.0)
        self.assertAlmostEqual(result.shoulder_line_tilt_degrees, 0.0)

    def test_upper_body_pose_needs_no_hip_landmarks(self) -> None:
        """Calculate both posture values from an image cropped above the waist."""
        landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        landmarks[7] = Landmark(0.5, 0.0)
        landmarks[8] = Landmark(0.5, 0.0)
        landmarks[11] = Landmark(0.0, 1.0)
        landmarks[12] = Landmark(1.0, 2.0)
        result = analyze_posture(landmarks)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.neck_tilt_degrees, 0.0)
        self.assertAlmostEqual(result.shoulder_line_tilt_degrees, 45.0)

    def test_unreliable_landmark_is_rejected(self) -> None:
        """Return no angles when an essential pose landmark is unreliable."""
        landmarks = [Landmark(0.0, 0.0) for _ in range(33)]
        landmarks[11].visibility = 0.1
        self.assertIsNone(analyze_posture(landmarks))

    def test_landmark_below_configured_confidence_threshold_is_rejected(self) -> None:
        """Reject face/shoulder landmarks below the configured visibility cutoff."""
        landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        landmarks[8].visibility = 0.64
        self.assertIsNone(analyze_posture(landmarks))

    def test_metric_filter_smooths_last_three_samples(self) -> None:
        """Average metrics over the configured recent sample window."""
        metric_filter = PostureMetricFilter(window_size=3)
        metric_filter.add(PostureAngles(10.0, 0.0, 0.10, 0.50), observed_at=0.0)
        metric_filter.add(PostureAngles(13.0, 0.0, 0.13, 0.53), observed_at=1.0)
        filtered = metric_filter.add(
            PostureAngles(16.0, 0.0, 0.16, 0.56), observed_at=2.0
        )
        self.assertIsNotNone(filtered)
        assert filtered is not None
        self.assertAlmostEqual(filtered.neck_tilt_degrees, 13.0)
        self.assertAlmostEqual(filtered.forward_head_ratio, 0.13)
        self.assertAlmostEqual(filtered.normalized_head_height, 0.53)

    def test_metric_filter_ignores_single_frame_outlier(self) -> None:
        """Skip an implausible isolated jump and accept a repeated stable new pose."""
        metric_filter = PostureMetricFilter(window_size=1)
        baseline = PostureAngles(10.0, 0.0, 0.10, 0.50)
        metric_filter.add(baseline, observed_at=0.0)
        outlier = PostureAngles(90.0, 0.0, 0.10, 0.50)
        self.assertIsNone(metric_filter.add(outlier, observed_at=1.0))
        self.assertEqual(metric_filter.value(), baseline)
        self.assertEqual(metric_filter.add(outlier, observed_at=2.0), outlier)

    def test_forward_head_ratio_is_normalized_by_shoulder_width(self) -> None:
        """Measure ear-midpoint horizontal offset as a fraction of shoulder width."""
        world_landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        world_landmarks[7] = Landmark(0.5, 0.0)
        world_landmarks[8] = Landmark(0.5, 0.0)
        world_landmarks[11] = Landmark(0.0, 1.0)
        world_landmarks[12] = Landmark(1.0, 1.0)
        image_landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        image_landmarks[7] = Landmark(0.6, 0.0)
        image_landmarks[8] = Landmark(0.6, 0.0)
        image_landmarks[11] = Landmark(0.3, 1.0)
        image_landmarks[12] = Landmark(0.7, 1.0)

        result = analyze_posture(world_landmarks, image_landmarks, (640, 480))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.forward_head_ratio, 0.25)

    def test_forward_head_is_rejected_when_shoulder_width_is_zero(self) -> None:
        """Reject a degenerate image-space normalization denominator."""
        landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        image_landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        self.assertIsNone(analyze_posture(landmarks, image_landmarks, (640, 480)))

    def test_normalized_head_height_is_invariant_to_resolution_scale(self) -> None:
        """Keep head-height/shoulder-width stable when image dimensions double."""
        world_landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        image_landmarks = [Landmark(0.5, 0.5) for _ in range(13)]
        image_landmarks[7] = Landmark(0.5, 0.40)
        image_landmarks[8] = Landmark(0.5, 0.40)
        image_landmarks[11] = Landmark(0.3, 0.60)
        image_landmarks[12] = Landmark(0.7, 0.60)

        standard = analyze_posture(world_landmarks, image_landmarks, (640, 480))
        doubled = analyze_posture(world_landmarks, image_landmarks, (1280, 960))

        self.assertIsNotNone(standard)
        self.assertIsNotNone(doubled)
        assert standard is not None and doubled is not None
        self.assertAlmostEqual(standard.normalized_head_height, 0.75)
        self.assertAlmostEqual(
            standard.normalized_head_height, doubled.normalized_head_height
        )

    def test_only_positive_deviation_toward_poor_posture_triggers(self) -> None:
        """Trigger only when any signed metric exceeds its threshold positively."""
        self.assertTrue(posture_exceeds_threshold(13.0, 0.0))
        self.assertTrue(posture_exceeds_threshold(0.0, 9.0))
        self.assertTrue(posture_exceeds_threshold(0.0, 0.0, 0.061))
        self.assertFalse(posture_exceeds_threshold(12.0, 8.0))
        self.assertFalse(posture_exceeds_threshold(-30.0, -50.0, -1.0))

    def test_sitting_straighter_for_fifteen_seconds_stays_normal(self) -> None:
        """Do not start a bad-pose candidate for prolonged negative deviations."""
        manager = AlertManager()
        start = datetime.now().astimezone()
        for elapsed in range(16):
            deviation_is_bad = posture_exceeds_threshold(-18.0, -24.0, -0.15)
            manager.update(
                24.0,
                start + timedelta(seconds=elapsed),
                posture_bad=deviation_is_bad,
            )

        self.assertFalse(manager.posture_bad)
        self.assertFalse(manager.grace_active)
        self.assertEqual(manager.deviation_duration(start + timedelta(seconds=15)), 0.0)

    def test_head_height_threshold_is_independent(self) -> None:
        """Trigger on normalized vertical head drop using its configurable cutoff."""
        self.assertTrue(posture_exceeds_threshold(0.0, 0.0, 0.061))
        self.assertFalse(posture_exceeds_threshold(0.0, 0.0, 0.059))

    def test_metric_deltas_keep_degrees_and_percentage_points_separate(self) -> None:
        """Compare neck degrees and normalized forward-head position independently."""
        current = PostureAngles(25.0, 1.0, 0.22, 0.22)
        baseline = PostureAngles(10.0, 1.0, 0.10, 0.10)
        self.assertEqual(
            posture_metric_deviations(current, baseline), (15.0, 12.0, 0.12)
        )

    def test_deviation_uses_largest_absolute_angle(self) -> None:
        """Use the strongest neck or torso delta for thresholding."""
        measured = posture_deviation(PostureAngles(30.0, 4.0), PostureAngles(10.0, 15.0))
        self.assertEqual(measured, (20.0, -11.0, 20.0))


class AlertManagerTests(unittest.TestCase):
    """Check debounce, alert thresholds, and automatic recovery."""

    def test_short_movement_does_not_open_an_episode(self) -> None:
        """Discard a deviation that returns to baseline before debounce completes."""
        manager = AlertManager(minimum_bad_seconds=3.0, grace_period_seconds=0.0)
        start = datetime.now().astimezone()
        manager.update(20.0, start)
        manager.update(20.0, start + timedelta(seconds=2))
        update = manager.update(0.0, start + timedelta(seconds=2.5))
        self.assertIsNone(update.episode_started)
        self.assertIsNone(update.episode_ended)

    def test_soft_and_notification_alerts_are_one_shot(self) -> None:
        """Emit the visual and system alerts once at their configured delays."""
        manager = AlertManager(
            minimum_bad_seconds=3.0,
            soft_alert_seconds=15.0,
            notification_seconds=30.0,
            grace_period_seconds=0.0,
        )
        start = datetime.now().astimezone()
        manager.update(20.0, start)
        started = manager.update(20.0, start + timedelta(seconds=3))
        self.assertEqual(started.episode_started, start)
        self.assertFalse(started.soft_alert)
        soft_alert = manager.update(20.0, start + timedelta(seconds=15))
        self.assertTrue(soft_alert.soft_alert)
        notified = manager.update(20.0, start + timedelta(seconds=30))
        self.assertTrue(notified.notification)
        repeated = manager.update(20.0, start + timedelta(seconds=31))
        self.assertFalse(repeated.soft_alert)
        self.assertFalse(repeated.notification)

    def test_good_posture_closes_episode(self) -> None:
        """End a confirmed episode when deviation returns within tolerance."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=0.0)
        start = datetime.now().astimezone()
        manager.update(20.0, start)
        recovered_at = start + timedelta(seconds=5)
        update = manager.update(0.0, recovered_at)
        self.assertEqual(update.episode_ended, recovered_at)

    def test_explicit_multi_metric_decision_controls_episode(self) -> None:
        """Use the combined neck/head decision instead of comparing mixed units."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=0.0)
        started_at = datetime.now().astimezone()
        bad_update = manager.update(1.0, started_at, posture_bad=True)
        self.assertEqual(bad_update.episode_started, started_at)
        good_update = manager.update(50.0, started_at + timedelta(seconds=1), posture_bad=False)
        self.assertEqual(good_update.episode_ended, started_at + timedelta(seconds=1))

    def test_two_second_recovery_preserves_bad_posture_timer(self) -> None:
        """Continue elapsed bad time after a two-second recovery within grace."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=5.0)
        start = datetime.now().astimezone()
        manager.update(1.0, start, posture_bad=True)
        started = manager.update(1.0, start + timedelta(seconds=5), posture_bad=True)
        self.assertEqual(started.episode_started, start)
        manager.update(0.0, start + timedelta(seconds=10), posture_bad=False)
        self.assertTrue(manager.grace_active)
        self.assertEqual(
            manager.grace_remaining(start + timedelta(seconds=12)), 3.0
        )

        resumed = manager.update(1.0, start + timedelta(seconds=12), posture_bad=True)

        self.assertIsNone(resumed.episode_ended)
        self.assertTrue(manager.posture_bad)
        self.assertFalse(manager.grace_active)
        self.assertEqual(manager.deviation_duration(start + timedelta(seconds=12)), 12.0)

    def test_skipped_observation_pauses_episode_clock(self) -> None:
        """Exclude low-confidence/outlier frame gaps from accumulated posture time."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=0.0)
        start = datetime.now().astimezone()
        manager.update(1.0, start, posture_bad=True)
        manager.pause_timers(start + timedelta(seconds=4))
        self.assertEqual(
            manager.deviation_duration(start + timedelta(seconds=8)), 4.0
        )
        manager.update(1.0, start + timedelta(seconds=8), posture_bad=True)
        self.assertEqual(
            manager.deviation_duration(start + timedelta(seconds=8)), 4.0
        )

    def test_brief_bad_posture_does_not_switch_normal_state(self) -> None:
        """Ignore a short bad-pose spike while confirmed posture is normal."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=5.0)
        start = datetime.now().astimezone()
        manager.update(1.0, start, posture_bad=True)
        self.assertFalse(manager.posture_bad)
        self.assertTrue(manager.grace_active)
        manager.update(0.0, start + timedelta(seconds=1), posture_bad=False)
        self.assertFalse(manager.posture_bad)
        self.assertFalse(manager.grace_active)

    def test_five_seconds_of_good_posture_ends_episode_at_recovery_start(self) -> None:
        """End an active episode only after the complete good-state grace period."""
        manager = AlertManager(minimum_bad_seconds=0.0, grace_period_seconds=5.0)
        start = datetime.now().astimezone()
        manager.update(1.0, start, posture_bad=True)
        manager.update(1.0, start + timedelta(seconds=5), posture_bad=True)
        recovery_started = start + timedelta(seconds=10)
        manager.update(0.0, recovery_started, posture_bad=False)
        self.assertTrue(manager.posture_bad)
        update = manager.update(0.0, recovery_started + timedelta(seconds=5), posture_bad=False)
        self.assertEqual(update.episode_ended, recovery_started)
        self.assertFalse(manager.posture_bad)

    def test_deviation_duration_drives_visible_timer(self) -> None:
        """Report elapsed seconds for the UI timer while deviation continues."""
        manager = AlertManager()
        started_at = datetime.now().astimezone()
        manager.update(20.0, started_at)
        self.assertEqual(
            manager.deviation_duration(started_at + timedelta(seconds=7)), 7.0
        )
        manager.reset()
        self.assertEqual(manager.deviation_duration(started_at), 0.0)


class DatabaseTests(unittest.TestCase):
    """Check that episode duration is persisted and summarized locally."""

    def test_full_day_episode_counts_as_full_day(self) -> None:
        """Clip and total episode time against the requested local day."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = PostureDatabase(Path(temporary_directory) / "history.sqlite3")
            now = datetime.now().astimezone()
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            episode_id = database.start_episode(midnight)
            database.end_episode(episode_id, now)
            self.assertAlmostEqual(database.daily_poor_posture_percent(now=now), 100.0)
            database.close()

    def test_daily_episodes_returns_local_time_intervals(self) -> None:
        """List today's episode start and end values for the statistics window."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = PostureDatabase(Path(temporary_directory) / "history.sqlite3")
            now = datetime.now().astimezone()
            started_at = now - timedelta(minutes=4)
            ended_at = now - timedelta(minutes=2)
            episode_id = database.start_episode(started_at)
            database.end_episode(episode_id, ended_at)
            episodes = database.daily_episodes(now=now)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0], (started_at, ended_at))
            database.close()


class CalibrationTests(unittest.TestCase):
    """Check upper-body calibration and rejection of the old hip-based baseline."""

    def test_calibration_saves_shoulder_line_baseline_without_hips(self) -> None:
        """Calculate and persist baseline angles from the upper-body analyzer."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "calibration.json"
            calibration = calibration_from_angles(
                PostureAngles(0.0, 45.0, 0.2, 1.15)
            )
            save_calibration(calibration, calibration_path)

            self.assertAlmostEqual(calibration.shoulder_line_tilt_degrees, 45.0)
            self.assertAlmostEqual(calibration.forward_head_ratio, 0.2)
            self.assertAlmostEqual(calibration.normalized_head_height, 1.15)
            self.assertEqual(load_calibration(calibration_path), calibration)

    def test_missing_pose_calibration_asks_for_retry(self) -> None:
        """Use the requested retry message when no posture angles were detected."""
        with self.assertRaisesRegex(ValueError, "Не удалось распознать позу"):
            calibration_from_angles(None)

    def test_old_hip_based_baseline_requires_recalibration(self) -> None:
        """Do not interpret a previous shoulders-to-hips angle as shoulder tilt."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_file = Path(temporary_directory) / "calibration.json"
            calibration_file.write_text(
                '{"neck_tilt_degrees": 8.0, "shoulder_tilt_degrees": 12.0, '
                '"photo_name": "old.jpg"}',
                encoding="utf-8",
            )
            self.assertIsNone(load_calibration(calibration_file))


class LiveCalibrationUiTests(unittest.TestCase):
    """Check that calibration countdown uses frames without blocking Tk."""

    def make_window_stub(self) -> SpineUpWindow:
        """Build the small mocked UI state needed by calibration state methods.

        Returns:
            An uninitialized SpineUpWindow with mocked overlay controls.
        """
        window = SpineUpWindow.__new__(SpineUpWindow)
        window.calibration_mode = True
        window.calibration_in_progress = False
        window.calibration_countdown_deadline = None
        window.calibration_retry_until = None
        window.calibration_confirmation_until = None
        window.calibration_attempt_id = 3
        window.analysis_thread = None
        window.calibration_banner = Mock()
        window.calibration_countdown = Mock()
        window.calibration_banner_text = Mock()
        window.calibration_countdown_text = Mock()
        window._start_analysis = Mock()
        window._set_status = Mock()
        return window

    def test_countdown_submits_current_frame_to_worker(self) -> None:
        """Capture the latest frame after three seconds without a blocking wait."""
        window = self.make_window_stub()
        frame = np.zeros((16, 24, 3), dtype=np.uint8)
        with patch("ui.time.monotonic", side_effect=(10.0, 13.01)):
            window._advance_calibration(frame)
            self.assertEqual(window.calibration_countdown_deadline, 13.0)
            window._advance_calibration(frame)
        self.assertTrue(window.calibration_in_progress)
        window._start_analysis.assert_called_once()
        submitted_frame = window._start_analysis.call_args.args[0]
        np.testing.assert_array_equal(submitted_frame, frame)
        self.assertTrue(window._start_analysis.call_args.kwargs["is_calibration"])
        self.assertEqual(window._start_analysis.call_args.kwargs["calibration_id"], 3)

    def test_failed_capture_shows_retry_and_keeps_calibration_mode(self) -> None:
        """Return an unsuccessful pose capture to the countdown instead of a dialog."""
        window = self.make_window_stub()
        window.calibration_in_progress = True
        result = AnalysisResult(
            detection=None,
            angles=None,
            error="Не удалось распознать позу, попробуй ещё раз",
            is_calibration=True,
            calibration_id=3,
        )
        with patch("ui.time.monotonic", return_value=20.0):
            window._apply_calibration_result(result)
        self.assertTrue(window.calibration_mode)
        self.assertFalse(window.calibration_in_progress)
        self.assertEqual(window.calibration_retry_until, 21.5)
        window.calibration_banner_text.set.assert_called_with(
            "Не удалось распознать позу, попробуй ещё раз"
        )

    def test_success_saves_baseline_and_resumes_monitoring(self) -> None:
        """Persist a valid capture and show a short confirmation over the video."""
        window = self.make_window_stub()
        window.calibration_in_progress = True
        window.calibration = None
        window.latest_analysis_time = 100.0
        window.latest_deviation = None
        window.latest_angle_deltas = None
        window.latest_debug_note = None
        window.metric_filter = PostureMetricFilter()
        window.deviation_text = Mock()
        window.timer_text = Mock()
        window.calibration_text = Mock()
        window.pause_button_text = Mock()
        calibration = calibration_from_angles(PostureAngles(5.0, 2.0))
        result = AnalysisResult(
            detection=None,
            angles=calibration.angles,
            is_calibration=True,
            calibration=calibration,
            calibration_id=3,
        )
        with patch("ui.time.monotonic", return_value=50.0), patch(
            "ui.save_calibration"
        ) as save:
            window._apply_calibration_result(result)
        save.assert_called_once_with(calibration)
        self.assertEqual(window.calibration, calibration)
        self.assertFalse(window.calibration_mode)
        self.assertTrue(window.monitoring)
        self.assertEqual(window.calibration_confirmation_until, 52.0)
        window.calibration_banner_text.set.assert_called_with("Калибровка сохранена ✅")

    def test_stale_attempt_cannot_replace_baseline(self) -> None:
        """Ignore an old worker result after calibration has been restarted."""
        window = self.make_window_stub()
        window.calibration_attempt_id = 4
        window.calibration = None
        result = AnalysisResult(
            detection=None,
            angles=None,
            error="Не удалось распознать позу, попробуй ещё раз",
            is_calibration=True,
            calibration_id=3,
        )
        with patch("ui.save_calibration") as save:
            window._apply_analysis_result(result)
        save.assert_not_called()

    def test_timer_shows_grace_countdown_while_state_is_pending(self) -> None:
        """Show a live grace countdown without changing confirmed posture state."""
        window = SpineUpWindow.__new__(SpineUpWindow)
        window.monitoring = True
        window.latest_deviation = 1.0
        window.timer_text = Mock()
        window.alert_manager = Mock()
        window.alert_manager.grace_active = True
        window.alert_manager.posture_bad = False
        window.alert_manager.grace_remaining.return_value = 3.2

        window._update_live_timer()

        window.timer_text.set.assert_called_once_with("Проверка... 4 сек")

    def test_low_confidence_pose_frame_is_skipped_without_state_update(self) -> None:
        """Leave status and posture timers untouched when hand occlusion lowers confidence."""
        window = SpineUpWindow.__new__(SpineUpWindow)
        window.calibration_mode = False
        window.monitoring = True
        window.calibration = calibration_from_angles(PostureAngles(10.0, 0.0))
        window.latest_landmarks = [Landmark(0.5, 0.5) for _ in range(13)]
        window.latest_deviation = 4.0
        window.latest_posture_bad = False
        window.latest_debug_note = None
        window.alert_manager = Mock()
        window.deviation_text = Mock()
        window._set_status = Mock()
        detection = Mock()
        detection.image_landmarks = window.latest_landmarks
        result = AnalysisResult(detection=detection, angles=None)

        window._apply_analysis_result(result)

        self.assertIsNone(window.latest_landmarks)
        self.assertEqual(window.latest_deviation, 4.0)
        self.assertFalse(window.latest_posture_bad)
        self.assertEqual(
            window.latest_debug_note,
            "НИЗКАЯ УВЕРЕННОСТЬ / КАДР ПРОПУЩЕН",
        )
        window.alert_manager.pause_timers.assert_called_once()
        window.alert_manager.update.assert_not_called()
        window._set_status.assert_not_called()


class ResourceOptimizationUiTests(unittest.TestCase):
    """Check rendering and pose inference are independently throttled."""

    def make_video_window(self, window_state: str, focused: bool) -> SpineUpWindow:
        """Build a small Tk window stub for a single scheduled video tick.

        Args:
            window_state: Tk window state, such as normal or iconic.
            focused: Whether the window reports display focus.

        Returns:
            An uninitialized SpineUpWindow with mocked frame-loop dependencies.
        """
        window = SpineUpWindow.__new__(SpineUpWindow)
        window.closed = False
        window.root = Mock()
        window.root.state.return_value = window_state
        window.root.focus_displayof.return_value = object() if focused else None
        window.root.after = Mock()
        window.capture = Mock()
        window.capture.read.return_value = (True, np.zeros((360, 480, 3), dtype=np.uint8))
        window._drain_analysis_results = Mock()
        window._update_cpu_meter = Mock()
        window._render_frame = Mock()
        window._advance_calibration = Mock()
        window._update_live_timer = Mock()
        window._try_open_camera = Mock()
        window.monitoring = False
        window.calibration_mode = False
        window.calibration = None
        window.latest_analysis_time = 0.0
        window.analysis_thread = None
        window._start_analysis = Mock()
        window.last_frame = None
        return window

    def test_minimized_window_skips_preview_render(self) -> None:
        """Do not render PIL/Tk images while minimized and poll camera slowly."""
        window = self.make_video_window("iconic", focused=False)
        with patch("ui.time.monotonic", return_value=10.0):
            window._video_tick()
        window._render_frame.assert_not_called()
        window.root.after.assert_called_once_with(250, window._video_tick)

    def test_pose_inference_is_throttled_across_frequent_video_ticks(self) -> None:
        """Read/render frames frequently but launch pose inference at most once per second."""
        window = self.make_video_window("normal", focused=True)
        window.monitoring = True
        window.calibration = Mock()
        window.latest_analysis_time = 100.0
        with patch("ui.time.monotonic", side_effect=(101.2, 101.8)):
            window._video_tick()
            window._video_tick()
        self.assertEqual(window.capture.read.call_count, 2)
        self.assertEqual(window._render_frame.call_count, 2)
        window._start_analysis.assert_called_once()

class SkeletonRenderingTests(unittest.TestCase):
    """Check that normalized landmarks are visibly drawn on camera frames."""

    def test_skeleton_overlay_draws_landmarks_and_connections(self) -> None:
        """Draw the configured pose graph without MediaPipe's legacy drawing API."""
        landmarks = [Landmark(0.5, 0.5) for _ in range(33)]
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = draw_pose_skeleton(frame, landmarks)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertGreater(len(SKELETON_CONNECTIONS), 0)

    def test_debug_overlay_includes_live_baseline_and_delta_values(self) -> None:
        """Expose current, baseline, and delta neck/head metrics in the frame corner."""
        window = SpineUpWindow.__new__(SpineUpWindow)
        window.calibration = calibration_from_angles(PostureAngles(10.0, 0.0, 0.1))
        window.latest_neck_angle = 25.0
        window.latest_forward_head_ratio = 0.2
        window.latest_cpu_percent = 2.0
        window.latest_debug_note = None
        window.latest_landmarks = [Landmark(0.0, 0.0) for _ in range(13)]
        window.latest_landmarks[11] = Landmark(0.3, 1.0)
        window.latest_landmarks[12] = Landmark(0.7, 1.0)
        window.latest_normalized_head_height = 0.82
        window.calibration = calibration_from_angles(PostureAngles(10.0, 0.0, 0.1, 0.70))

        lines = window._debug_overlay_lines(640, 480)

        self.assertIn("25.0 deg", lines[0])
        self.assertIn("base 10.0", lines[0])
        self.assertIn("delta +15.0", lines[0])
        self.assertIn("+51.2px", lines[1])
        self.assertIn("+20.0%", lines[1])
        self.assertIn("+25.6px", lines[2])
        self.assertIn("+10.0%", lines[2])
        self.assertIn("0.820", lines[3])
        self.assertIn("base 0.700", lines[3])
        self.assertIn("delta +0.120", lines[3])


if __name__ == "__main__":
    unittest.main()

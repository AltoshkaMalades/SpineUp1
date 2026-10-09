"""Tkinter interface for live SpineUp posture monitoring."""

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from typing import Sequence

import cv2
import numpy as np
import psutil
from PIL import Image, ImageTk
from plyer import notification
from break_window import BreakWindow
from exercises import EXERCISES
from autostart import disable_autostart, enable_autostart, is_autostart_enabled

from alert_manager import AlertManager, AlertUpdate
from calibration import Calibration, calibration_from_angles, save_calibration
from config import (
    APP_NAME,
    CAMERA_INDEX,
    CPU_METER_UPDATE_SECONDS,
    ICON_ICO_PATH,
    ICON_PNG_PATH,
    NOTIFICATION_SECONDS,
    POSTURE_DEVIATION_THRESHOLD_DEGREES,
    PREVIEW_ACTIVE_INTERVAL_MS,
    PREVIEW_BACKGROUND_INTERVAL_MS,
    SAMPLE_INTERVAL_SECONDS,
    __version__,
)
from database import PostureDatabase
from pose_detector import PoseDetection, PoseDetector, open_camera
from posture_analyzer import (
    PostureAngles,
    PostureMetricFilter,
    analyze_posture,
    posture_exceeds_threshold,
    posture_metric_deviations,
)


SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (7, 8),
    (7, 11),
    (8, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
)


@dataclass(frozen=True)
class AnalysisResult:
    """One background pose-analysis result for a captured frame."""

    detection: PoseDetection | None
    angles: PostureAngles | None
    error: str | None = None
    is_calibration: bool = False
    calibration: Calibration | None = None
    calibration_id: int | None = None


def draw_pose_skeleton(
    frame: np.ndarray,
    landmarks: Sequence[object] | None,
    debug_lines: Sequence[str] = (),
) -> np.ndarray:
    """Draw visible pose landmarks and their body connections on a BGR frame.

    Args:
        frame: OpenCV frame to annotate in BGR color order.
        landmarks: MediaPipe normalized image landmarks, or None if no pose exists.
        debug_lines: Small diagnostic strings drawn in the top-left corner.

    Returns:
        The annotated frame, modified in place.
    """
    frame_height, frame_width = frame.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    if landmarks is not None:
        for index, landmark in enumerate(landmarks):
            if getattr(landmark, "visibility", 1.0) < 0.3:
                continue
            if getattr(landmark, "presence", 1.0) < 0.3:
                continue
            point = (
                int(float(landmark.x) * frame_width),
                int(float(landmark.y) * frame_height),
            )
            points[index] = point

        for first_index, second_index in SKELETON_CONNECTIONS:
            first_point = points.get(first_index)
            second_point = points.get(second_index)
            if first_point is not None and second_point is not None:
                cv2.line(frame, first_point, second_point, (60, 220, 110), 3, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(frame, point, 4, (40, 230, 255), -1, cv2.LINE_AA)

    for line_index, text in enumerate(debug_lines):
        y_position = 22 + line_index * 18
        cv2.putText(
            frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


class SpineUpWindow:
    """Own the Tk window, camera preview, posture state, and local history."""

    def __init__(
        self,
        detector: PoseDetector,
        database: PostureDatabase,
        calibration: Calibration | None,
        camera_index: int = CAMERA_INDEX,
    ) -> None:
        """Create the SpineUp window and initialize camera/UI state.

        Args:
            detector: Loaded MediaPipe Tasks detector.
            database: Open local posture history database.
            calibration: Previously saved user baseline, if available.
            camera_index: OpenCV camera device index.

        Returns:
            None.
        """
        self.detector = detector
        self.database = database
        self.calibration = calibration
        self.camera_index = camera_index
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{__version__}")
        self._window_icon = tk.PhotoImage(file=str(ICON_PNG_PATH))
        self.root.iconbitmap(default=str(ICON_ICO_PATH))
        self.root.iconphoto(True, self._window_icon)
        self.root.geometry("1080x760")
        self.root.minsize(760, 560)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_app_menu()

        self.alert_manager = AlertManager(
            deviation_threshold=POSTURE_DEVIATION_THRESHOLD_DEGREES
        )
        self.break_window: BreakWindow | None = None
        self.monitoring = calibration is not None
        self.capture: cv2.VideoCapture | None = None
        self.last_frame: np.ndarray | None = None
        self.latest_landmarks: Sequence[object] | None = None
        self.latest_deviation: float | None = None
        self.latest_angle_deltas: tuple[float, float] | None = None
        self.latest_neck_angle: float | None = None
        self.latest_forward_head_ratio: float | None = None
        self.latest_normalized_head_height: float | None = None
        self.latest_posture_bad = False
        self.metric_filter = PostureMetricFilter()
        if calibration is not None:
            self.metric_filter.reset(calibration.angles)
        self.latest_debug_note: str | None = None
        self.latest_cpu_percent = 0.0
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        self._last_cpu_update = time.monotonic()
        self.latest_analysis_time = 0.0
        self.next_camera_retry = 0.0
        self.episode_id: int | None = None
        self.closed = False
        self.calibration_mode = False
        self.calibration_in_progress = False
        self.calibration_countdown_deadline: float | None = None
        self.calibration_retry_until: float | None = None
        self.calibration_confirmation_until: float | None = None
        self.calibration_attempt_id = 0
        self.analysis_results: queue.Queue[AnalysisResult] = queue.Queue(maxsize=2)
        self.analysis_thread: threading.Thread | None = None

        self.status_text = tk.StringVar()
        self.deviation_text = tk.StringVar(value="Отклонение: --")
        self.timer_text = tk.StringVar(value="Длительность: --")
        self.calibration_text = tk.StringVar()
        self.pause_button_text = tk.StringVar()
        self._build_widgets()
        self._set_status("paused" if not self.monitoring else "normal")
        self._update_calibration_label()
        self._update_pause_button()
        self.root.after(0, self._video_tick)
        if self.calibration is None:
            self.root.after(300, self.calibrate)

    def run(self) -> None:
        """Enter Tkinter's non-blocking event-driven main loop.

        Returns:
            None.
        """
        self.root.mainloop()

    def _build_app_menu(self) -> None:
        """Add the startup option to the window used by the main entry point."""
        menu_bar = tk.Menu(self.root)
        settings_menu = tk.Menu(menu_bar, tearoff=False)
        self._autostart_var = tk.BooleanVar(value=self._read_autostart_enabled())
        settings_menu.add_checkbutton(
            label="Launch at startup",
            variable=self._autostart_var,
            command=self._toggle_autostart,
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label=f"{APP_NAME} v{__version__}", state="disabled"
        )
        menu_bar.add_cascade(label="Settings", menu=settings_menu)
        self.root.config(menu=menu_bar)

    def _read_autostart_enabled(self) -> bool:
        """Read startup state without allowing registry errors to stop launch."""
        try:
            return is_autostart_enabled()
        except Exception as error:
            self._notify_autostart_error(error)
            return False

    def _toggle_autostart(self) -> None:
        """Apply the startup menu selection and restore it if registration fails."""
        enabled = self._autostart_var.get()
        try:
            if enabled:
                enable_autostart()
            else:
                disable_autostart()
        except Exception as error:
            self._autostart_var.set(not enabled)
            self._notify_autostart_error(error)

    def _notify_autostart_error(self, error: Exception) -> None:
        """Show registry errors as a desktop notification, without crashing."""
        message = f"Could not update Launch at startup: {error}"
        try:
            notification.notify(title=APP_NAME, message=message, app_name=APP_NAME)
        except Exception:
            print(f"[SpineUp] {message}")

    def _build_widgets(self) -> None:
        """Create the status banner, video frame, and action buttons.

        Returns:
            None.
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        status_panel = tk.Frame(self.root, bg="#218C55", padx=18, pady=14)
        status_panel.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        status_panel.columnconfigure(0, weight=1)
        self.status_label = tk.Label(
            status_panel,
            textvariable=self.status_text,
            bg=status_panel["bg"],
            fg="white",
            font=("Segoe UI", 20, "bold"),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.deviation_label = tk.Label(
            status_panel,
            textvariable=self.deviation_text,
            bg=status_panel["bg"],
            fg="white",
            font=("Segoe UI", 15),
            anchor="w",
        )
        self.deviation_label.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.timer_label = tk.Label(
            status_panel,
            textvariable=self.timer_text,
            bg=status_panel["bg"],
            fg="white",
            font=("Segoe UI", 13),
            anchor="e",
        )
        self.timer_label.grid(row=1, column=1, sticky="e", padx=(12, 0), pady=(5, 0))

        video_shell = tk.Frame(self.root, bg="#218C55", padx=7, pady=7)
        video_shell.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        video_shell.rowconfigure(0, weight=1)
        video_shell.columnconfigure(0, weight=1)
        self.video_label = tk.Label(
            video_shell,
            text="Подключение к камере...",
            bg="#202729",
            fg="white",
            font=("Segoe UI", 16),
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Configure>", self._on_video_resize)
        self.video_shell = video_shell
        self.calibration_banner_text = tk.StringVar()
        self.calibration_banner = tk.Label(
            video_shell,
            textvariable=self.calibration_banner_text,
            bg="#202729",
            fg="white",
            font=("Segoe UI", 17, "bold"),
            padx=14,
            pady=8,
            wraplength=720,
        )
        self.calibration_countdown_text = tk.StringVar()
        self.calibration_countdown = tk.Label(
            video_shell,
            textvariable=self.calibration_countdown_text,
            bg="#202729",
            fg="white",
            font=("Segoe UI", 48, "bold"),
            padx=18,
            pady=6,
        )

        footer = tk.Frame(self.root, padx=16, pady=8)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(4, weight=1)
        tk.Button(
            footer,
            text="Калибровка",
            command=self.calibrate,
            font=("Segoe UI", 12),
            padx=12,
            pady=7,
        ).grid(row=0, column=0, padx=(0, 8))
        tk.Button(
            footer,
            text="Перерыв 1 мин",
            command=self.start_posture_break,
            font=("Segoe UI", 12),
            padx=12,
            pady=7,
        ).grid(row=0, column=3, padx=8)
        self.pause_button = tk.Button(
            footer,
            textvariable=self.pause_button_text,
            command=self.toggle_monitoring,
            font=("Segoe UI", 12),
            padx=12,
            pady=7,
        )
        self.pause_button.grid(row=0, column=1, padx=8)
        tk.Button(
            footer,
            text="Статистика за сегодня",
            command=self.show_statistics,
            font=("Segoe UI", 12),
            padx=12,
            pady=7,
        ).grid(row=0, column=2, padx=8)
        tk.Label(
            footer,
            textvariable=self.calibration_text,
            font=("Segoe UI", 10),
            fg="#4A5558",
            anchor="e",
        ).grid(row=0, column=4, sticky="e")

    def _set_status(self, state: str) -> None:
        """Set the large status label and high-visibility color frame.

        Args:
            state: One of normal, warning, alert, paused, missing, or error.

        Returns:
            None.
        """
        styles = {
            "normal": ("✅ Осанка в норме", "#218C55", "white"),
            "warning": ("⚠️ Начинаешь сутулиться", "#D99A00", "#171717"),
            "alert": ("🔴 Выпрямись!", "#C62828", "white"),
            "paused": ("Мониторинг на паузе", "#697579", "white"),
            "missing": ("Поза не распознана — сядьте перед камерой", "#53656B", "white"),
            "error": ("Камера недоступна", "#8E3434", "white"),
            "calibration": ("Сначала выполните калибровку", "#53656B", "white"),
            "calibrating": ("Калибровка по камере", "#176B78", "white"),
        }
        text, color, foreground = styles[state]
        self.status_text.set(text)
        self.status_label.configure(bg=color, fg=foreground)
        self.deviation_label.configure(bg=color, fg=foreground)
        self.timer_label.configure(bg=color, fg=foreground)
        self.video_shell.configure(bg=color)

    def _update_calibration_label(self) -> None:
        """Show whether a personal camera baseline has been calibrated.

        Returns:
            None.
        """
        if self.calibration is None:
            self.calibration_text.set("Эталон: не задан")
        else:
            self.calibration_text.set("Эталон: живая камера")

    def _update_pause_button(self) -> None:
        """Set the action label for the current monitoring state.

        Returns:
            None.
        """
        self.pause_button_text.set(
            "Пауза мониторинга" if self.monitoring else "Продолжить мониторинг"
        )

    def _video_tick(self) -> None:
        """Read and display camera frames, scheduling the next UI update.

        Returns:
            None.
        """
        if self.closed:
            return
        self._drain_analysis_results()
        preview_visible = self._preview_visible()
        self._update_cpu_meter()
        if self.capture is None:
            self._try_open_camera()
        if self.capture is not None:
            success, frame = self.capture.read()
            if success and frame is not None:
                self.last_frame = frame
                now = time.monotonic()
                if (
                    self.monitoring
                    and not self.calibration_mode
                    and self.calibration is not None
                    and now - self.latest_analysis_time >= SAMPLE_INTERVAL_SECONDS
                    and (self.analysis_thread is None or not self.analysis_thread.is_alive())
                ):
                    self.latest_analysis_time = now
                    self._start_analysis(frame.copy())
                if preview_visible:
                    self._render_frame(frame)
                self._advance_calibration(frame)
            else:
                self._camera_failed("Камера не вернула кадр.")
        self._update_live_timer()
        interval_ms = (
            PREVIEW_ACTIVE_INTERVAL_MS
            if preview_visible
            else PREVIEW_BACKGROUND_INTERVAL_MS
        )
        self.root.after(interval_ms, self._video_tick)

    def _preview_visible(self) -> bool:
        """Return whether the window is focused and not minimized.

        Returns:
            True when rendering is useful to the user; False in the background.
        """
        try:
            return (
                self.root.state() != "iconic"
                and self.root.focus_displayof() is not None
            )
        except tk.TclError:
            return False

    def _update_cpu_meter(self) -> None:
        """Update process CPU usage at the configured low diagnostic rate.

        Returns:
            None.
        """
        now = time.monotonic()
        if now - self._last_cpu_update < CPU_METER_UPDATE_SECONDS:
            return
        try:
            self.latest_cpu_percent = self._process.cpu_percent(interval=None)
        except psutil.Error:
            self.latest_cpu_percent = 0.0
        self._last_cpu_update = now

    def _try_open_camera(self) -> None:
        """Try to open the camera, retrying periodically after a failure.

        Returns:
            None.
        """
        now = time.monotonic()
        if now < self.next_camera_retry:
            return
        try:
            self.capture = open_camera(self.camera_index)
            self.video_label.configure(text="")
            if self.monitoring and self.calibration is not None:
                self._set_status("normal")
            print(f"[SpineUp] Камера {self.camera_index} подключена.")
        except (RuntimeError, cv2.error) as error:
            self.next_camera_retry = now + 2.0
            self._set_status("error")
            self.video_label.configure(text=f"Камера недоступна\n{error}")

    def _camera_failed(self, message: str) -> None:
        """Release a camera that stopped producing frames and show an error.

        Args:
            message: Human-readable camera failure details.

        Returns:
            None.
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.next_camera_retry = time.monotonic() + 2.0
        self._set_status("error")
        self.video_label.configure(text=message, image="")
        self.photo_image = None

    def _start_analysis(
        self,
        frame: np.ndarray,
        is_calibration: bool = False,
        calibration_id: int | None = None,
    ) -> None:
        """Run pose inference on a copied frame outside Tk's UI thread.

        Args:
            frame: Captured BGR frame reserved for this analysis sample.
            is_calibration: Whether this inference belongs to calibration.
            calibration_id: Current attempt id, used to discard stale results.

        Returns:
            None.
        """
        self.analysis_thread = threading.Thread(
            target=self._analyze_frame,
            args=(frame, is_calibration, calibration_id),
            name="SpineUpCalibration" if is_calibration else "SpineUpPoseInference",
            daemon=True,
        )
        self.analysis_thread.start()

    def _analyze_frame(
        self,
        frame: np.ndarray,
        is_calibration: bool = False,
        calibration_id: int | None = None,
    ) -> None:
        """Infer pose and posture angles, then post the result to Tk's queue.

        Args:
            frame: One sampled BGR camera frame.
            is_calibration: Whether the frame is a calibration capture.
            calibration_id: Attempt id associated with the capture.

        Returns:
            None.
        """
        try:
            detection = self.detector.detect_pose(frame)
            angles = (
                analyze_posture(
                    detection.world_landmarks,
                    detection.image_landmarks,
                    detection.image_size,
                )
                if detection is not None
                else None
            )
            calibration = (
                calibration_from_angles(angles)
                if is_calibration
                else None
            )
            error = (
                "Не удалось распознать позу, попробуй ещё раз"
                if is_calibration and calibration is None
                else None
            )
            result = AnalysisResult(
                detection=detection,
                angles=angles,
                error=error,
                is_calibration=is_calibration,
                calibration=calibration,
                calibration_id=calibration_id,
            )
        except (ValueError, RuntimeError, cv2.error):
            result = AnalysisResult(
                detection=None,
                angles=None,
                error="Не удалось распознать позу, попробуй ещё раз",
                is_calibration=is_calibration,
                calibration_id=calibration_id,
            )
        try:
            self.analysis_results.put_nowait(result)
        except queue.Full:
            try:
                self.analysis_results.get_nowait()
            except queue.Empty:
                pass
            self.analysis_results.put_nowait(result)

    def _drain_analysis_results(self) -> None:
        """Apply the newest completed analysis on Tk's main thread.

        Returns:
            None.
        """
        newest_result: AnalysisResult | None = None
        while True:
            try:
                newest_result = self.analysis_results.get_nowait()
            except queue.Empty:
                break
        if newest_result is not None:
            self._apply_analysis_result(newest_result)

    def _apply_analysis_result(self, result: AnalysisResult) -> None:
        """Update pose overlay, deviation labels, and alert/database state.

        Args:
            result: Completed worker inference result.

        Returns:
            None.
        """
        if result.is_calibration:
            if result.calibration_id == self.calibration_attempt_id:
                self._apply_calibration_result(result)
            return
        if self.calibration_mode:
            return
        if result.error is not None:
            print(f"[SpineUp] Ошибка анализа кадра: {result.error}")
            self._set_status("error")
            return
        if result.detection is None:
            self.latest_landmarks = None
            self.latest_debug_note = "ПОЗА НЕ НАЙДЕНА / КАДР ПРОПУЩЕН"
            if self.monitoring:
                update = self.alert_manager.update(None, datetime.now().astimezone())
                self._apply_alert_update(update)
                self._set_status("missing")
            return

        self.latest_landmarks = result.detection.image_landmarks
        if result.angles is None or self.calibration is None:
            self.latest_landmarks = None
            self.latest_debug_note = "НИЗКАЯ УВЕРЕННОСТЬ / КАДР ПРОПУЩЕН"
            if self.monitoring:
                self.alert_manager.pause_timers(datetime.now().astimezone())
            return

        filtered_angles = self.metric_filter.add(result.angles, time.monotonic())
        if filtered_angles is None:
            self.latest_debug_note = "РЕЗКИЙ СКАЧОК / КАДР ПРОПУЩЕН"
            if self.monitoring:
                self.alert_manager.pause_timers(datetime.now().astimezone())
            return
        self.latest_debug_note = None

        (
            neck_delta,
            forward_head_delta_percent,
            head_height_delta,
        ) = posture_metric_deviations(
            filtered_angles, self.calibration.angles
        )
        self.latest_neck_angle = filtered_angles.neck_tilt_degrees
        self.latest_forward_head_ratio = filtered_angles.forward_head_ratio
        self.latest_normalized_head_height = filtered_angles.normalized_head_height
        self.latest_angle_deltas = (
            neck_delta,
            forward_head_delta_percent,
            head_height_delta,
        )
        self.latest_posture_bad = posture_exceeds_threshold(
            neck_delta,
            forward_head_delta_percent,
            head_height_delta,
        )
        self.latest_deviation = max(
            neck_delta,
            forward_head_delta_percent,
            head_height_delta,
            0.0,
        )
        self.deviation_text.set(
            f"Шея Δ {neck_delta:+.1f}° | Вперёд Δ {forward_head_delta_percent:+.1f}% "
            f"| Высота Δ {head_height_delta:+.3f}"
        )
        if not self.monitoring:
            self._set_status("paused")
            return

        now = datetime.now().astimezone()
        update = self.alert_manager.update(
            self.latest_deviation,
            now,
            posture_bad=self.latest_posture_bad,
        )
        self._apply_alert_update(update)
        if not self.alert_manager.posture_bad:
            self._set_status("normal")
        elif self.alert_manager.deviation_duration(now) >= NOTIFICATION_SECONDS:
            self._set_status("alert")
        else:
            self._set_status("warning")

    def _apply_calibration_result(self, result: AnalysisResult) -> None:
        """Apply a calibration result or schedule an automatic retry.

        Args:
            result: Pose inference and calibration calculation result.

        Returns:
            None.
        """
        self.calibration_in_progress = False
        if result.calibration_id != self.calibration_attempt_id or not self.calibration_mode:
            return
        if result.error is not None or result.calibration is None:
            self.calibration_mode = True
            self.calibration_retry_until = time.monotonic() + 1.5
            self.calibration_countdown_deadline = None
            self.calibration_banner_text.set(
                "Не удалось распознать позу, попробуй ещё раз"
            )
            self.calibration_countdown_text.set("")
            self._set_status("calibrating")
            return

        try:
            save_calibration(result.calibration)
        except OSError as error:
            self.calibration_mode = True
            self.calibration_retry_until = time.monotonic() + 1.5
            self.calibration_countdown_deadline = None
            self.calibration_banner_text.set(f"Не удалось сохранить калибровку: {error}")
            self._set_status("calibrating")
            return

        self.calibration = result.calibration
        self.metric_filter.reset(result.calibration.angles)
        self.calibration_mode = False
        self.calibration_retry_until = None
        self.calibration_countdown_deadline = None
        self.calibration_confirmation_until = time.monotonic() + 2.0
        if result.detection is not None:
            self.latest_landmarks = result.detection.image_landmarks
        self.latest_neck_angle = result.calibration.neck_tilt_degrees
        self.latest_forward_head_ratio = result.calibration.forward_head_ratio
        self.latest_normalized_head_height = result.calibration.normalized_head_height
        self.latest_posture_bad = False
        self.latest_debug_note = None
        self.latest_deviation = 0.0
        self.latest_angle_deltas = (0.0, 0.0, 0.0)
        self.deviation_text.set("Отклонение: 0°")
        self.timer_text.set("Длительность: --")
        self.monitoring = True
        self.latest_analysis_time = 0.0
        self._update_calibration_label()
        self._update_pause_button()
        self.calibration_banner_text.set("Калибровка сохранена ✅")
        self.calibration_countdown_text.set("")
        self._set_status("normal")

    def _advance_calibration(self, frame: np.ndarray) -> None:
        """Advance countdown and submit the current frame for calibration.

        Args:
            frame: The latest frame already read by the regular camera loop.

        Returns:
            None.
        """
        now = time.monotonic()
        if not self.calibration_mode:
            if (
                self.calibration_confirmation_until is not None
                and now < self.calibration_confirmation_until
            ):
                self.calibration_banner.place(
                    relx=0.5, rely=0.04, anchor="n", relwidth=0.9
                )
                self.calibration_countdown.place_forget()
            else:
                self.calibration_confirmation_until = None
                self.calibration_banner.place_forget()
                self.calibration_countdown.place_forget()
            return

        self.calibration_banner.place(relx=0.5, rely=0.04, anchor="n", relwidth=0.9)
        self.calibration_countdown.place(relx=0.5, rely=0.48, anchor="center")
        if self.calibration_in_progress:
            self.calibration_banner_text.set("Проверяю позу...")
            self.calibration_countdown_text.set("…")
            return
        if self.calibration_retry_until is not None:
            if now < self.calibration_retry_until:
                self.calibration_countdown_text.set("")
                return
            self.calibration_retry_until = None
            self.calibration_countdown_deadline = None

        self.calibration_banner_text.set(
            "Сядь в привычной нормальной позе, не выпрямляйся специально. "
            "Захват через 3 секунды..."
        )
        if self.calibration_countdown_deadline is None:
            self.calibration_countdown_deadline = now + 3.0
        remaining = self.calibration_countdown_deadline - now
        if remaining <= 0.0:
            if self.analysis_thread is not None and self.analysis_thread.is_alive():
                self.calibration_countdown_text.set("...")
                return
            self.calibration_in_progress = True
            self.calibration_countdown_text.set("…")
            self._start_analysis(
                frame.copy(),
                is_calibration=True,
                calibration_id=self.calibration_attempt_id,
            )
            return
        self.calibration_countdown_text.set(str(int(np.ceil(remaining))))

    def _apply_alert_update(self, update: AlertUpdate) -> None:
        """Persist episode transitions and show the delayed system notification.

        Args:
            update: State transitions produced by AlertManager.

        Returns:
            None.
        """
        if update.episode_started is not None:
            self.episode_id = self.database.start_episode(update.episode_started)
        if update.episode_ended is not None and self.episode_id is not None:
            self.database.end_episode(self.episode_id, update.episode_ended)
            self.episode_id = None
        if update.notification:
            try:
                notification.notify(
                    title=APP_NAME,
                    message="Выпрямись и расслабь плечи",
                    app_name=APP_NAME,
                    timeout=8,
                )
            except Exception as error:
                print(f"[SpineUp] Не удалось показать уведомление: {error}")

    def _update_live_timer(self) -> None:
        """Refresh elapsed deviation time while the UI is displaying video.

        Returns:
            None.
        """
        if not self.monitoring or self.latest_deviation is None:
            self.timer_text.set("Длительность: --")
            return
        if self.alert_manager.grace_active:
            remaining = int(
                np.ceil(self.alert_manager.grace_remaining(datetime.now().astimezone()))
            )
            if self.alert_manager.posture_bad:
                elapsed = int(
                    self.alert_manager.deviation_duration(datetime.now().astimezone())
                )
                self.timer_text.set(
                    f"Сутулость: {elapsed} сек · проверка нормы: {remaining} сек"
                )
            else:
                self.timer_text.set(f"Проверка... {remaining} сек")
            return
        if not self.alert_manager.posture_bad:
            self.timer_text.set("Длительность: --")
            return
        elapsed = int(self.alert_manager.deviation_duration(datetime.now().astimezone()))
        self.timer_text.set(f"Длительность: {elapsed} сек")
        if elapsed >= NOTIFICATION_SECONDS:
            self._set_status("alert")

    def _render_frame(self, frame: np.ndarray) -> None:
        """Render the current webcam frame with the latest skeleton overlay.

        Args:
            frame: Current BGR webcam frame.

        Returns:
            None.
        """
        annotated = draw_pose_skeleton(
            frame,
            self.latest_landmarks,
            self._debug_overlay_lines(frame.shape[1], frame.shape[0]),
        )
        available_width = max(self.video_label.winfo_width(), 320)
        available_height = max(self.video_label.winfo_height(), 240)
        scale = min(available_width / annotated.shape[1], available_height / annotated.shape[0])
        output_size = (
            max(1, int(annotated.shape[1] * scale)),
            max(1, int(annotated.shape[0] * scale)),
        )
        resized = cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.video_label.configure(image=self.photo_image, text="")

    def _debug_overlay_lines(
        self, frame_width: int, frame_height: int
    ) -> tuple[str, ...]:
        """Format current, baseline, and delta posture metrics for the video.

        Args:
            frame_width: Current camera frame width in pixels.
            frame_height: Current camera frame height in pixels.

        Returns:
            Compact lines suitable for drawing in the video corner.
        """
        if self.calibration is None:
            return (
                f"CPU: {self.latest_cpu_percent:.1f}%",
                *( (self.latest_debug_note,) if self.latest_debug_note else () ),
                "DEBUG: baseline not calibrated",
            )

        baseline = self.calibration.angles
        if (
            self.latest_neck_angle is None
            or self.latest_forward_head_ratio is None
            or self.latest_normalized_head_height is None
        ):
            return (
                f"Neck: -- | base {baseline.neck_tilt_degrees:.1f} deg | delta --",
                f"Head X: -- | base {baseline.forward_head_ratio * 100:+.1f}% | delta --",
                f"Head height: -- | base {baseline.normalized_head_height:.3f} | delta --",
                f"CPU: {self.latest_cpu_percent:.1f}%",
                *((self.latest_debug_note,) if self.latest_debug_note else ()),
            )

        neck_delta = self.latest_neck_angle - baseline.neck_tilt_degrees
        forward_delta_percent = (
            self.latest_forward_head_ratio - baseline.forward_head_ratio
        ) * 100.0
        head_height_delta = (
            self.latest_normalized_head_height - baseline.normalized_head_height
        )
        shoulder_width_pixels = 0.0
        if self.latest_landmarks is not None and len(self.latest_landmarks) > 12:
            shoulder_width_pixels = float(
                np.hypot(
                    (self.latest_landmarks[11].x - self.latest_landmarks[12].x)
                    * frame_width,
                    (self.latest_landmarks[11].y - self.latest_landmarks[12].y)
                    * frame_height,
                )
            )
        current_offset_pixels = self.latest_forward_head_ratio * shoulder_width_pixels
        baseline_offset_pixels = baseline.forward_head_ratio * shoulder_width_pixels
        delta_offset_pixels = current_offset_pixels - baseline_offset_pixels
        return (
            f"Neck: {self.latest_neck_angle:.1f} deg | base "
            f"{baseline.neck_tilt_degrees:.1f} | delta {neck_delta:+.1f}",
            f"Head X: {current_offset_pixels:+.1f}px / "
            f"{self.latest_forward_head_ratio * 100:+.1f}% shoulders",
            f"Head base: {baseline_offset_pixels:+.1f}px / "
            f"{baseline.forward_head_ratio * 100:+.1f}% | delta "
            f"{delta_offset_pixels:+.1f}px / {forward_delta_percent:+.1f}%",
            f"Head height: {self.latest_normalized_head_height:.3f} | "
            f"base {baseline.normalized_head_height:.3f} | "
            f"delta {head_height_delta:+.3f}",
            f"CPU: {self.latest_cpu_percent:.1f}%",
            *((self.latest_debug_note,) if self.latest_debug_note else ()),
        )

    def _on_video_resize(self, _event: tk.Event) -> None:
        """Allow the next camera tick to resize its image to the new viewport.

        Args:
            _event: Tk configure event.

        Returns:
            None.
        """
        del _event
        if (
            self.last_frame is not None
            and not self.closed
            and self._preview_visible()
        ):
            self._render_frame(self.last_frame.copy())

    def calibrate(self) -> None:
        """Start or restart a live-camera calibration countdown.

        Returns:
            None.
        """
        self.calibration_attempt_id += 1
        update = self.alert_manager.reset(datetime.now().astimezone())
        self._apply_alert_update(update)
        self.monitoring = False
        self.calibration_mode = True
        self.calibration_in_progress = False
        self.calibration_countdown_deadline = None
        self.calibration_retry_until = None
        self.calibration_confirmation_until = None
        self.calibration_banner_text.set(
            "Сядь в привычной нормальной позе, не выпрямляйся специально. "
            "Захват через 3 секунды..."
        )
        self.calibration_countdown_text.set("3")
        self.timer_text.set("Длительность: --")
        self._update_pause_button()
        self._set_status("calibrating")

    def toggle_monitoring(self) -> None:
        """Pause or resume posture analysis without stopping the live preview.

        Returns:
            None.
        """
        if self.calibration_mode:
            self.calibration_attempt_id += 1
            self.calibration_mode = False
            self.calibration_in_progress = False
            self.calibration_retry_until = None
            self.calibration_countdown_deadline = None
            self.monitoring = self.calibration is not None
            self._update_pause_button()
            self._set_status("normal" if self.monitoring else "calibration")
            return
        if self.calibration is None:
            messagebox.showinfo(
                APP_NAME,
                "Сначала выполните калибровку по камере.",
                parent=self.root,
            )
            return
        if self.monitoring:
            update = self.alert_manager.reset(datetime.now().astimezone())
            self._apply_alert_update(update)
            self.monitoring = False
            self._set_status("paused")
        else:
            self.monitoring = True
            self.latest_analysis_time = 0.0
            self._set_status("normal")
        self._update_pause_button()
    def start_posture_break(self) -> None:
            """Open the 1-minute posture exercises window."""
            if self.break_window is not None:
                return
            self.break_window = BreakWindow(self.root, EXERCISES, self._on_break_finished)
    
    def _on_break_finished(self, completed: bool) -> None:
            """Called when the break window closes."""
            self.break_window = None
            if completed:
                self.database.record_break(datetime.now().astimezone())

    def show_statistics(self) -> None:
        """Open a window with today's good-posture percentage and episodes.

        Returns:
            None.
        """
        poor_percent = self.database.daily_poor_posture_percent()
        episodes = self.database.daily_episodes()
        window = tk.Toplevel(self.root)
        window.title("Статистика за сегодня")
        window.geometry("520x380")
        window.minsize(420, 280)
        window.transient(self.root)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)

        good_percent = max(0.0, 100.0 - poor_percent)
        tk.Label(
            window,
            text=f"Хорошая осанка сегодня: {good_percent:.1f}%",
            font=("Segoe UI", 18, "bold"),
            pady=16,
        ).grid(row=0, column=0, sticky="w", padx=18)
        tk.Label(
            window,
            text=f"Эпизодов сутулости: {len(episodes)}",
            font=("Segoe UI", 12),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 8))

        episode_list = tk.Listbox(window, font=("Segoe UI", 11), activestyle="none")
        episode_list.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        if not episodes:
            episode_list.insert(tk.END, "Сегодня эпизодов не зафиксировано.")
        for started_at, ended_at in episodes:
            end_label = ended_at.strftime("%H:%M:%S") if ended_at else "сейчас"
            episode_list.insert(
                tk.END,
                f"{started_at.strftime('%H:%M:%S')} — {end_label}",
            )

    def close(self) -> None:
        """Stop Tk callbacks and release camera, detector, and database resources.

        Returns:
            None.
        """
        if self.closed:
            return
        self.closed = True
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        worker = self.analysis_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        update = self.alert_manager.reset(datetime.now().astimezone())
        self._apply_alert_update(update)
        self.detector.close()
        self.database.close()
        self.root.destroy()
    
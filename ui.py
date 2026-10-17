"""Tkinter interface for live SpineUp posture monitoring."""

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
import tkinter as tk
from typing import Callable, Sequence

import cv2
import numpy as np
import psutil
import pystray
from PIL import Image, ImageTk
from plyer import notification
from break_window import BreakPopup, BreakWindow
from exercises import EXERCISES, Exercise
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
    BREAK_SNOOZE_SECONDS,
    SITTING_BREAK_SECONDS,
    THEME,
    __version__,
)
from database import PostureDatabase
from demo_data import load_demo_data
from pose_detector import PoseDetection, PoseDetector, open_camera
from statistics_window import StatisticsWindow
from storage import EventType, SessionStorage
from posture_analyzer import (
    PostureAngles,
    PostureMetricFilter,
    analyze_posture,
    posture_exceeds_threshold,
    posture_metric_deviations,
)
from window_chrome import create_title_bar, make_frameless, show_themed_message


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
                cv2.line(
                    frame,
                    first_point,
                    second_point,
                    THEME["skeleton_line_bgr"],
                    3,
                    cv2.LINE_AA,
                )
        for point in points.values():
            cv2.circle(frame, point, 4, THEME["skeleton_point_bgr"], -1, cv2.LINE_AA)

    for line_index, text in enumerate(debug_lines):
        y_position = 22 + line_index * 18
        cv2.putText(
            frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            THEME["debug_overlay_bgr"],
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
        storage: SessionStorage | None = None,
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
        self.storage = storage
        self.calibration = calibration
        self.camera_index = camera_index
        self.root = tk.Tk()
        self.root.configure(bg=THEME["border"])
        self.root.title(f"{APP_NAME} v{__version__}")
        make_frameless(self.root, taskbar=True)
        self._window_icon = tk.PhotoImage(file=str(ICON_PNG_PATH))
        self._title_icon = self._window_icon.subsample(
            max((self._window_icon.width() + 27) // 28, 1),
            max((self._window_icon.height() + 27) // 28, 1),
        )
        self.root.iconbitmap(default=str(ICON_ICO_PATH))
        self.root.iconphoto(True, self._window_icon)
        self.root.geometry("1080x760")
        self.root.minsize(820, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.alert_manager = AlertManager(
            deviation_threshold=POSTURE_DEVIATION_THRESHOLD_DEGREES
        )
        self.break_window: BreakWindow | None = None
        self.break_popup: BreakPopup | None = None
        self._break_exercise_index = -1
        self._sitting_elapsed = 0.0
        self._last_sitting_tick = time.monotonic()
        self._break_snooze_until: float | None = None
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
        self._session_id: int | None = None
        self._statistics_window: StatisticsWindow | None = None
        self._tray_icon: pystray.Icon | None = None
        self.closed = False
        self.calibration_mode = False
        self.calibration_in_progress = False
        self.calibration_countdown_deadline: float | None = None
        self.calibration_retry_until: float | None = None
        self.calibration_confirmation_until: float | None = None
        self.calibration_attempt_id = 0
        self.analysis_results: queue.Queue[AnalysisResult] = queue.Queue(maxsize=2)
        self.analysis_thread: threading.Thread | None = None
        self._settings_window: tk.Toplevel | None = None
        self.debug_info_var = tk.BooleanVar(value=False)

        self.status_text = tk.StringVar()
        self.deviation_text = tk.StringVar(value="Отклонение: --")
        self.timer_text = tk.StringVar(value="Длительность: --")
        self.calibration_text = tk.StringVar()
        self.pause_button_text = tk.StringVar()
        self._build_widgets()
        self._update_pause_button()
        self._set_status("paused" if not self.monitoring else "normal")
        if self.monitoring:
            self._start_monitoring_session()
        self._update_calibration_label()
        self.root.after_idle(lambda: self.root.update_idletasks())
        self._start_tray_icon()
        self.root.after(0, self._video_tick)
        self.root.after(1000, self._check_break_reminder)
        if self.calibration is None:
            self.root.after(300, self.calibrate)

    def run(self) -> None:
        """Enter Tkinter's non-blocking event-driven main loop.

        Returns:
            None.
        """
        self.root.mainloop()

    def _build_app_menu(self) -> None:
        """Initialize the launch-at-startup setting for the Settings panel."""
        self._autostart_var = tk.BooleanVar(value=self._read_autostart_enabled())
        self._settings_window = None

    def _start_tray_icon(self) -> None:
        """Expose monitoring controls and statistics in the Windows tray."""
        try:
            with Image.open(ICON_PNG_PATH) as source:
                icon_image = source.convert("RGBA")
            self._tray_icon = pystray.Icon(
                APP_NAME,
                icon_image,
                APP_NAME,
                menu=pystray.Menu(
                    pystray.MenuItem("Open SpineUp", self._tray_open_window),
                    pystray.MenuItem(
                        self._tray_monitoring_label, self._tray_toggle_monitoring
                    ),
                    pystray.MenuItem("Statistics", self._tray_show_statistics),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Exit", self._tray_close),
                ),
            )
            self._tray_icon.run_detached()
        except Exception as error:
            self._tray_icon = None
            print(f"[SpineUp] Не удалось запустить значок в трее: {error}")

    def _dispatch_tray_action(self, action: Callable[[], None]) -> None:
        try:
            self.root.after(0, action)
        except tk.TclError as error:
            print(f"[SpineUp] Не удалось передать действие из трея: {error}")

    def _tray_monitoring_label(self, _item: object) -> str:
        del _item
        return "Pause monitoring" if self.monitoring else "Start monitoring"

    def _tray_open_window(self, _icon: object, _item: object) -> None:
        del _icon, _item
        self._dispatch_tray_action(self._show_main_window)

    def _tray_toggle_monitoring(self, _icon: object, _item: object) -> None:
        del _icon, _item
        self._dispatch_tray_action(self.toggle_monitoring)

    def _tray_show_statistics(self, _icon: object, _item: object) -> None:
        del _icon, _item
        self._dispatch_tray_action(self.show_statistics)

    def _tray_close(self, _icon: object, _item: object) -> None:
        del _icon, _item
        self._dispatch_tray_action(self.close)

    def _show_main_window(self) -> None:
        self.root.deiconify()
        self.root.after_idle(lambda: self.root.focus_force())

    def _start_monitoring_session(self) -> None:
        storage = getattr(self, "storage", None)
        if storage is None:
            return
        try:
            self._session_id = storage.start_session()
        except (OSError, sqlite3.Error) as error:
            self._session_id = None
            print(f"[SpineUp] Не удалось открыть сессию мониторинга: {error}")

    def _end_monitoring_session(self) -> None:
        storage = getattr(self, "storage", None)
        if storage is None:
            return
        try:
            storage.end_session()
        except (OSError, sqlite3.Error) as error:
            print(f"[SpineUp] Не удалось закрыть сессию мониторинга: {error}")
        finally:
            self._session_id = None

    def _log_monitoring_event(self, event_type: EventType) -> None:
        storage = getattr(self, "storage", None)
        if storage is None:
            return
        try:
            storage.log_event(
                event_type, session_id=getattr(self, "_session_id", None)
            )
        except (OSError, sqlite3.Error) as error:
            print(f"[SpineUp] Не удалось сохранить событие {event_type}: {error}")

    def _log_monitoring_frame(self, is_slouching: bool) -> None:
        storage = getattr(self, "storage", None)
        if storage is None:
            return
        try:
            storage.log_frame(is_slouching)
        except (OSError, sqlite3.Error) as error:
            print(f"[SpineUp] Не удалось сохранить кадр мониторинга: {error}")

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
        self._build_app_menu()
        title_bar = create_title_bar(
            self.root,
            "SpineUp",
            self._title_icon,
            on_close=self.close,
        )
        title_bar.pack(fill="x", padx=1, pady=(1, 0))
        self.app_frame = tk.Frame(self.root, bg=THEME["background"])
        self.app_frame.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.app_frame.columnconfigure(0, weight=1)
        self.app_frame.rowconfigure(2, weight=1)

        actions = tk.Frame(self.app_frame, bg=THEME["background"], padx=16, pady=12)
        actions.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            actions.columnconfigure(column, weight=1, uniform="action")
        button_specs = (
            ("Калибровка", self.calibrate, False),
            ("Пауза мониторинга", self.toggle_monitoring, True),
            ("Статистика", self.show_statistics, False),
            ("Перерыв 1 мин", self.start_posture_break, False),
            ("Настройки", self.show_settings, False),
        )
        self.pause_button = None
        for column, (text, command, primary) in enumerate(button_specs):
            options = {
                "text": text,
                "command": command,
                "font": (THEME["font"], 11, "bold"),
                "bg": THEME["accent"] if primary else THEME["surface"],
                "fg": THEME["background"] if primary else THEME["text"],
                "activebackground": THEME["accent_hover"] if primary else THEME["surface_hover"],
                "activeforeground": THEME["background"] if primary else THEME["text"],
                "relief": "flat",
                "borderwidth": 0,
                "cursor": "hand2",
                "padx": 11,
                "pady": 9,
            }
            button = tk.Button(actions, **options)
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
            normal_bg = options["bg"]
            hover_bg = THEME["accent_hover"] if primary else THEME["surface_hover"]
            button.bind("<Enter>", lambda _event, widget=button, color=hover_bg: widget.configure(bg=color))
            button.bind("<Leave>", lambda _event, widget=button, color=normal_bg: widget.configure(bg=color))
            if column == 1:
                self.pause_button = button
                button.configure(textvariable=self.pause_button_text)

        status_panel = tk.Frame(
            self.app_frame,
            bg=THEME["good"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
            padx=18,
            pady=14,
        )
        status_panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        status_panel.columnconfigure(0, weight=1)
        self.status_label = tk.Label(
            status_panel,
            textvariable=self.status_text,
            bg=THEME["good"],
            fg=THEME["good_text"],
            font=(THEME["font"], 20, "bold"),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.deviation_label = tk.Label(
            status_panel,
            textvariable=self.deviation_text,
            bg=THEME["good"],
            fg=THEME["good_text"],
            font=(THEME["font"], 12),
            anchor="w",
        )
        self.deviation_label.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.timer_label = tk.Label(
            status_panel,
            textvariable=self.timer_text,
            bg=THEME["good"],
            fg=THEME["muted"],
            font=(THEME["font"], 11),
            anchor="e",
        )
        self.timer_label.grid(row=1, column=1, sticky="e", padx=(12, 0), pady=(5, 0))

        video_shell = tk.Frame(
            self.app_frame,
            bg=THEME["border"],
            padx=1,
            pady=1,
        )
        video_shell.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 10))
        video_shell.rowconfigure(0, weight=1)
        video_shell.columnconfigure(0, weight=1)
        self.video_label = tk.Label(
            video_shell,
            text="Подключение к камере...",
            bg=THEME["surface"],
            fg=THEME["muted"],
            font=(THEME["font"], 16),
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Configure>", self._on_video_resize)
        self.video_shell = video_shell
        self.calibration_banner_text = tk.StringVar()
        self.calibration_banner = tk.Label(
            video_shell,
            textvariable=self.calibration_banner_text,
            bg=THEME["surface"],
            fg=THEME["text"],
            font=(THEME["font"], 17, "bold"),
            padx=14,
            pady=8,
            wraplength=720,
        )
        self.calibration_countdown_text = tk.StringVar()
        self.calibration_countdown = tk.Label(
            video_shell,
            textvariable=self.calibration_countdown_text,
            bg=THEME["surface"],
            fg=THEME["accent"],
            font=(THEME["font"], 48, "bold"),
            padx=18,
            pady=6,
        )

        footer = tk.Frame(self.app_frame, bg=THEME["background"], padx=16, pady=6)
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        tk.Label(
            footer,
            textvariable=self.calibration_text,
            bg=THEME["background"],
            font=(THEME["font"], 10),
            fg=THEME["muted"],
            anchor="e",
        ).grid(row=0, column=0, sticky="e")

    def show_settings(self) -> None:
        """Open the visual and startup settings panel."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.deiconify()
            self._settings_window.lift()
            return
        window = tk.Toplevel(self.root)
        self._settings_window = window
        window.title("Настройки SpineUp")
        window.configure(bg=THEME["border"])
        window.geometry("460x360")
        window.transient(self.root)
        make_frameless(window)
        create_title_bar(
            window, "Настройки", self._title_icon, window.destroy
        ).pack(fill="x", padx=1, pady=(1, 0))
        panel = tk.Frame(window, bg=THEME["background"], padx=16, pady=12)
        panel.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        tk.Label(
            panel,
            text="Параметры приложения",
            bg=THEME["background"],
            fg=THEME["text"],
            font=(THEME["font"], 15, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        tk.Checkbutton(
            panel,
            text="Показывать отладочную информацию на видео",
            variable=self.debug_info_var,
            command=self._toggle_debug_info,
            bg=THEME["background"],
            fg=THEME["text"],
            activebackground=THEME["background"],
            activeforeground=THEME["text"],
            selectcolor=THEME["surface"],
            font=(THEME["font"], 10),
            anchor="w",
        ).pack(fill="x", pady=5)
        tk.Checkbutton(
            panel,
            text="Запускать SpineUp вместе с Windows",
            variable=self._autostart_var,
            command=self._toggle_autostart,
            bg=THEME["background"],
            fg=THEME["text"],
            activebackground=THEME["background"],
            activeforeground=THEME["text"],
            selectcolor=THEME["surface"],
            font=(THEME["font"], 10),
            anchor="w",
        ).pack(fill="x", pady=5)
        for label, command in (
            ("Проверить напоминание о перерыве", self.test_break_popup),
            ("Загрузить демо-данные", self._load_demo_data_from_menu),
        ):
            tk.Button(
                panel,
                text=label,
                command=command,
                bg=THEME["surface"],
                fg=THEME["text"],
                activebackground=THEME["surface_hover"],
                activeforeground=THEME["text"],
                relief="flat",
                padx=12,
                pady=8,
                font=(THEME["font"], 10),
                cursor="hand2",
            ).pack(fill="x", pady=(8, 0))
        window.protocol("WM_DELETE_WINDOW", window.destroy)

    def _toggle_debug_info(self) -> None:
        if self.last_frame is not None and self._preview_visible():
            self._render_frame(self.last_frame.copy())

    def _set_status(self, state: str) -> None:
        """Set the large status label and high-visibility color frame.

        Args:
            state: One of normal, warning, alert, paused, missing, or error.

        Returns:
            None.
        """
        styles = {
            "normal": ("✓ Отличная осанка", THEME["good"], THEME["good_text"], THEME["border"]),
            "warning": ("Выпрямись!", THEME["coral"], THEME["coral_text"], THEME["coral_border"]),
            "alert": ("Выпрямись!", THEME["coral"], THEME["coral_text"], THEME["coral_border"]),
            "paused": ("Мониторинг на паузе", THEME["surface"], THEME["text"], THEME["border"]),
            "missing": ("Поза не распознана — сядьте перед камерой", THEME["surface"], THEME["text"], THEME["border"]),
            "error": ("Камера недоступна", THEME["surface"], THEME["text"], THEME["border"]),
            "calibration": ("Сначала выполните калибровку", THEME["surface"], THEME["text"], THEME["border"]),
            "calibrating": ("Калибровка по камере", THEME["surface"], THEME["text"], THEME["border"]),
        }
        text, color, foreground, border = styles[state]
        self.status_text.set(text)
        self.status_label.configure(bg=color, fg=foreground)
        self.deviation_label.configure(bg=color, fg=foreground)
        self.timer_label.configure(bg=color, fg=THEME["muted"] if state == "normal" else foreground)
        self.status_label.master.configure(bg=color, highlightbackground=border)
        self.timer_label.configure(bg=color)
        self.video_shell.configure(bg=THEME["coral_border"] if state in ("warning", "alert") else THEME["border"])

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
            "Пауза мониторинга" if self.monitoring else "Продолжить"
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

    def _check_break_reminder(self) -> None:
        """Count monitored time and open a reminder when its deadline arrives."""
        if self.closed:
            return
        now = time.monotonic()
        elapsed = max(now - self._last_sitting_tick, 0.0)
        self._last_sitting_tick = now
        if self.monitoring and self.break_popup is None:
            if self._break_snooze_until is not None:
                if now >= self._break_snooze_until:
                    self._break_snooze_until = None
                    self._fire_break_reminder()
            else:
                self._sitting_elapsed += elapsed
                if self._sitting_elapsed >= SITTING_BREAK_SECONDS:
                    self._fire_break_reminder()
        self.root.after(1000, self._check_break_reminder)

    def _fire_break_reminder(self) -> None:
        """Show the plain notification and then try to open the exercise popup."""
        self._sitting_elapsed = 0.0
        message = "You've been sitting for 60 minutes. Take a 1-minute posture break."
        try:
            notification.notify(title=APP_NAME, message=message, app_name=APP_NAME)
        except Exception as error:
            print(f"[SpineUp] Не удалось показать уведомление: {error}")
        self._show_break_popup()

    def _next_break_exercise(self) -> Exercise:
        """Advance to the next exercise so consecutive popups differ."""
        self._break_exercise_index = (self._break_exercise_index + 1) % len(EXERCISES)
        return EXERCISES[self._break_exercise_index]

    def _show_break_popup(self) -> None:
        """Open the compact popup without allowing display errors to escape."""
        if self.break_popup is not None:
            return
        try:
            self.break_popup = BreakPopup(
                self.root, self._next_break_exercise(), self._on_break_popup_finished
            )
            self._log_monitoring_event("break_started")
        except Exception as error:
            self.break_popup = None
            print(f"[SpineUp] Не удалось показать перерыв: {error}")

    def test_break_popup(self) -> None:
        """Open the reminder popup from Settings for a manual check."""
        self._show_break_popup()

    def _on_break_popup_finished(self, action: str) -> None:
        """Apply completion or snooze after the reminder popup closes."""
        self.break_popup = None
        now = time.monotonic()
        self._last_sitting_tick = now
        if action == "done":
            self._sitting_elapsed = 0.0
            self._break_snooze_until = None
            self._log_monitoring_event("break_completed")
            try:
                self.database.record_break(datetime.now().astimezone())
            except sqlite3.Error as error:
                print(f"[SpineUp] Не удалось сохранить перерыв: {error}")
        else:
            self._break_snooze_until = now + BREAK_SNOOZE_SECONDS

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

        self._log_monitoring_frame(self.latest_posture_bad)
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
        self._start_monitoring_session()
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
        if update.episode_ended is not None:
            self._log_monitoring_event("posture_corrected")
        if update.episode_ended is not None and self.episode_id is not None:
            self.database.end_episode(self.episode_id, update.episode_ended)
            self.episode_id = None
        if update.notification:
            self._log_monitoring_event("slouch_alert")
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
            self._debug_overlay_lines(frame.shape[1], frame.shape[0])
            if self.debug_info_var.get()
            else (),
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
        self._end_monitoring_session()
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
            if self.monitoring:
                self._start_monitoring_session()
            else:
                self._end_monitoring_session()
            self._update_pause_button()
            self._set_status("normal" if self.monitoring else "calibration")
            return
        if self.calibration is None:
            show_themed_message(
                self.root,
                APP_NAME,
                "Сначала выполните калибровку по камере.",
            )
            return
        if self.monitoring:
            update = self.alert_manager.reset(datetime.now().astimezone())
            self._apply_alert_update(update)
            self.monitoring = False
            self._end_monitoring_session()
            self._set_status("paused")
        else:
            self.monitoring = True
            self._start_monitoring_session()
            self.latest_analysis_time = 0.0
            self._set_status("normal")
        self._update_pause_button()
    def start_posture_break(self) -> None:
        """Open the 1-minute posture exercises window."""
        if self.break_window is not None:
            return
        self._log_monitoring_event("break_started")
        self.break_window = BreakWindow(self.root, EXERCISES, self._on_break_finished)
    
    def _on_break_finished(self, completed: bool) -> None:
        """Called when the break window closes."""
        self.break_window = None
        if completed:
            self._log_monitoring_event("break_completed")
            try:
                self.database.record_break(datetime.now().astimezone())
            except sqlite3.Error as error:
                print(f"[SpineUp] Не удалось сохранить завершённый перерыв: {error}")

    def show_statistics(self) -> None:
        """Open the local weekly statistics dashboard."""
        if (
            self._statistics_window is not None
            and self._statistics_window.window.winfo_exists()
        ):
            self._statistics_window.window.deiconify()
            self._statistics_window.window.lift()
            self._statistics_window.refresh()
            return
        self._statistics_window = StatisticsWindow(
            self.root, self.storage, self.load_demo_data
        )

    def load_demo_data(self) -> int:
        """Fill the dashboard with sample sessions without replacing user data."""
        if self.storage is None:
            raise RuntimeError("Local statistics storage is unavailable.")
        added = load_demo_data(self.storage)
        print(f"[SpineUp] Добавлено демо-сессий: {added}")
        return added

    def _load_demo_data_from_menu(self) -> None:
        try:
            added = self.load_demo_data()
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
            print(f"[SpineUp] Не удалось загрузить демо-данные: {error}")
            show_themed_message(
                self.root,
                APP_NAME,
                f"Не удалось загрузить демо-данные:\n{error}",
                "error",
            )
            return
        if (
            self._statistics_window is not None
            and self._statistics_window.window.winfo_exists()
        ):
            self._statistics_window.refresh()
        show_themed_message(self.root, APP_NAME, f"Добавлено демо-сессий: {added}")

    def close(self) -> None:
        """Stop Tk callbacks and release camera, detector, and database resources.

        Returns:
            None.
        """
        if self.closed:
            return
        self.closed = True
        if self.break_popup is not None:
            self.break_popup._finish("later")
            self.break_popup = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        worker = self.analysis_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        update = self.alert_manager.reset(datetime.now().astimezone())
        self._apply_alert_update(update)
        self._end_monitoring_session()
        if self._tray_icon is not None:
            try:
                self._tray_icon.stop()
            except Exception as error:
                print(f"[SpineUp] Не удалось закрыть значок в трее: {error}")
        self.detector.close()
        self.database.close()
        if self.storage is not None:
            try:
                self.storage.close()
            except (OSError, sqlite3.Error) as error:
                print(f"[SpineUp] Не удалось закрыть SQLite-хранилище: {error}")
        self.root.destroy()
    
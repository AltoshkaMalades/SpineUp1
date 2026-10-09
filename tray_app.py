"""Windows system-tray application and camera monitoring worker."""

import threading
from datetime import datetime
from typing import Protocol

import cv2
import pystray
from PIL import Image, ImageDraw
from plyer import notification

from alert_manager import AlertManager, AlertUpdate
from calibration import Calibration
from config import (
    APP_NAME,
    CAMERA_INDEX,
    FORWARD_HEAD_DEVIATION_THRESHOLD_PERCENT,
    NECK_ANGLE_DEVIATION_THRESHOLD_DEGREES,
    NORMALIZED_HEAD_HEIGHT_DEVIATION_THRESHOLD,
    POSTURE_DEVIATION_THRESHOLD_DEGREES,
    SAMPLE_INTERVAL_SECONDS,
    __version__,
    resource_path,
)
from autostart import disable_autostart, enable_autostart, is_autostart_enabled
from database import PostureDatabase
from pose_detector import PoseDetector, open_camera
from posture_analyzer import analyze_posture, posture_metric_deviations, posture_exceeds_threshold


class TrayIcon(Protocol):
    """Minimal tray-icon operation used by the legacy menu callback."""

    def stop(self) -> None:
        """Stop the tray icon event loop."""


class TrayApplication:
    """Coordinate monitoring, alerts, database, and the system tray menu."""

    def __init__(
        self,
        detector: PoseDetector,
        database: PostureDatabase,
        calibration: Calibration | None,
        camera_index: int = CAMERA_INDEX,
    ) -> None:
        """Initialize the tray UI and background worker state.

        Args:
            detector: Loaded local MediaPipe detector.
            database: Open local posture history database.
            calibration: Saved personal baseline, if available.
            camera_index: Camera device index to monitor.

        Returns:
            None.
        """
        self.detector = detector
        self.database = database
        self.calibration = calibration
        self.camera_index = camera_index
        self.alert_manager = AlertManager(
            deviation_threshold=POSTURE_DEVIATION_THRESHOLD_DEGREES
        )
        self._monitoring = threading.Event()
        self._shutdown = threading.Event()
        self._state_lock = threading.Lock()
        self._episode_id: int | None = None
        self._latest_status = (
            "Ожидание запуска" if calibration is not None else "Калибровка не выполнена"
        )
        self._worker = threading.Thread(
            target=self._monitor_camera, name="SpineUpCamera", daemon=True
        )
        self._icon = pystray.Icon(
            APP_NAME,
            self._make_icon("green"),
            APP_NAME,
            menu=pystray.Menu(
                pystray.MenuItem(self._monitoring_label, self._toggle_monitoring),
                pystray.MenuItem(
                    "Launch at startup",
                    self._toggle_autostart,
                    checked=self._autostart_checked,
                ),
                pystray.MenuItem("Показать статус", self._show_status),
                pystray.MenuItem("Статистика за сегодня", self._show_statistics),
                pystray.MenuItem(f"{APP_NAME} v{__version__}", None, enabled=False),
                pystray.MenuItem("Выход", self._quit),
            ),
        )

    def run(self) -> None:
        """Start the worker, begin monitoring when calibrated, and run the tray.

        Returns:
            None.
        """
        self._worker.start()
        if self.calibration is not None:
            self.start_monitoring()
        self._icon.run()
        self._shutdown.set()
        self._monitoring.clear()
        self._worker.join(timeout=5.0)
        self.detector.close()
        self.database.close()

    def start_monitoring(self) -> None:
        """Enable camera analysis when a valid personal baseline is available.

        Returns:
            None.
        """
        if self.calibration is None:
            self._show_message("Сначала выполните калибровку по фото правильной осанки.")
            return
        self._latest_status = "Запуск камеры..."
        self._monitoring.set()
        self._set_icon("green")

    def _monitoring_label(self, _item: object) -> str:
        """Return the current start/pause menu label.

        Args:
            _item: Menu item supplied by pystray.

        Returns:
            A localized action label.
        """
        del _item
        return "Пауза мониторинга" if self._monitoring.is_set() else "Начать мониторинг"

    def _toggle_monitoring(self, _icon: object, _item: object) -> None:
        """Toggle monitoring from the tray menu.

        Args:
            _icon: Tray icon supplied by pystray.
            _item: Menu item supplied by pystray.

        Returns:
            None.
        """
        del _icon, _item
        if self._monitoring.is_set():
            self._pause_monitoring()
        else:
            self.start_monitoring()

    def _autostart_checked(self, _item: object) -> bool:
        """Read the current startup setting for pystray's check mark."""
        try:
            return is_autostart_enabled()
        except Exception as error:
            self._notify_startup_error(error)
            return False

    def _toggle_autostart(self, _icon: object, _item: object) -> None:
        """Toggle Windows startup registration from the tray menu."""
        try:
            if is_autostart_enabled():
                disable_autostart()
            else:
                enable_autostart()
            self._icon.update_menu()
        except Exception as error:
            self._notify_startup_error(error)

    def _notify_startup_error(self, error: Exception) -> None:
        """Report a registry error without allowing it to stop the app."""
        message = f"Could not update Launch at startup: {error}"
        try:
            notification.notify(title=APP_NAME, message=message, app_name=APP_NAME)
        except Exception:
            print(f"[SpineUp] {message}")

    def _pause_monitoring(self) -> None:
        """Pause capture and close any episode currently being recorded.

        Returns:
            None.
        """
        self._monitoring.clear()
        with self._state_lock:
            update = self.alert_manager.reset(datetime.now().astimezone())
        self._apply_alert_update(update)
        self._latest_status = "Мониторинг на паузе"
        self._set_icon("gray")

    def _monitor_camera(self) -> None:
        """Read camera frames at a low rate and feed posture observations.

        Returns:
            None.
        """
        while not self._shutdown.is_set():
            if not self._monitoring.wait(timeout=0.25):
                continue
            capture: cv2.VideoCapture | None = None
            try:
                capture = open_camera(self.camera_index)
                self._latest_status = "Камера подключена"
                while self._monitoring.is_set() and not self._shutdown.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise RuntimeError("Камера не вернула кадр.")
                    self._process_frame(frame)
                    self._shutdown.wait(SAMPLE_INTERVAL_SECONDS)
            except (RuntimeError, cv2.error) as error:
                self._latest_status = f"Ошибка камеры: {error}"
                print(f"[SpineUp] {self._latest_status}")
                self._monitoring.clear()
                with self._state_lock:
                    update = self.alert_manager.reset(datetime.now().astimezone())
                self._apply_alert_update(update)
                self._set_icon("gray")
            finally:
                if capture is not None:
                    capture.release()

    def _process_frame(self, frame: object) -> None:
        """Analyze one captured frame and update alert state.

        Args:
            frame: BGR frame returned by OpenCV.

        Returns:
            None.
        """
        try:
            detection = self.detector.detect_pose(frame)
        except (ValueError, RuntimeError, cv2.error) as error:
            self._latest_status = f"Ошибка кадра: {error}"
            print(f"[SpineUp] {self._latest_status}")
            return
        observed_at = datetime.now().astimezone()
        if detection is None:
            self._latest_status = "Поза не распознана на кадре"
            with self._state_lock:
                update = self.alert_manager.update(None, observed_at)
            self._apply_alert_update(update)
            return
        angles = analyze_posture(
            detection.world_landmarks,
            detection.image_landmarks,
            detection.image_size,
        )
        if angles is None or self.calibration is None:
            self._latest_status = "Недостаточно видимых ориентиров"
            return

        neck_delta, forward_delta_percent, head_height_delta = posture_metric_deviations(
            angles,
            self.calibration.angles,
        )
        posture_bad = posture_exceeds_threshold(
            neck_delta,
            forward_delta_percent,
            head_height_delta,
            neck_threshold_degrees=NECK_ANGLE_DEVIATION_THRESHOLD_DEGREES,
            forward_head_threshold_percent=FORWARD_HEAD_DEVIATION_THRESHOLD_PERCENT,
            normalized_head_height_threshold=NORMALIZED_HEAD_HEIGHT_DEVIATION_THRESHOLD,
        )
        deviation = max(neck_delta, forward_delta_percent, head_height_delta, 0.0)
        self._latest_status = (
            f"Осанка в норме: шея {neck_delta:+.1f}°, голова "
            f"{forward_delta_percent:+.1f}%, высота {head_height_delta:+.3f}"
            if not posture_bad
            else f"Отклонение (шея {neck_delta:+.1f}°, "
            f"голова {forward_delta_percent:+.1f}%, высота {head_height_delta:+.3f})"
        )
        with self._state_lock:
            update = self.alert_manager.update(
                deviation, observed_at, posture_bad=posture_bad
            )
        self._apply_alert_update(update)

    def _apply_alert_update(self, update: AlertUpdate) -> None:
        """Persist episode transitions and present alerts to the user.

        Args:
            update: State transition returned by AlertManager.

        Returns:
            None.
        """
        if update.episode_started is not None:
            self._episode_id = self.database.start_episode(update.episode_started)
            print("[SpineUp] Зафиксирован эпизод сутулости.")
        if update.episode_ended is not None and self._episode_id is not None:
            self.database.end_episode(self._episode_id, update.episode_ended)
            self._episode_id = None
            print("[SpineUp] Эпизод сутулости завершён.")
        if update.soft_alert:
            self._set_icon("amber")
        elif update.episode_ended is not None:
            self._set_icon("green" if self._monitoring.is_set() else "gray")
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

    def _show_status(self, _icon: object, _item: object) -> None:
        """Display the latest monitoring status.

        Args:
            _icon: Tray icon supplied by pystray.
            _item: Menu item supplied by pystray.

        Returns:
            None.
        """
        del _icon, _item
        self._show_message(self._latest_status)

    def _show_statistics(self, _icon: object, _item: object) -> None:
        """Display the percentage of today's time recorded with poor posture.

        Args:
            _icon: Tray icon supplied by pystray.
            _item: Menu item supplied by pystray.

        Returns:
            None.
        """
        del _icon, _item
        percentage = self.database.daily_poor_posture_percent()
        message = f"Плохая осанка сегодня: {percentage:.1f}% времени"
        print(f"[SpineUp] {message}")
        self._show_message(message)

    def _quit(self, icon: TrayIcon, _item: object) -> None:
        """Stop the tray application and its monitoring worker.

        Args:
            icon: Tray icon supplied by pystray.
            _item: Menu item supplied by pystray.

        Returns:
            None.
        """
        del _item
        self._monitoring.clear()
        self._shutdown.set()
        with self._state_lock:
            update = self.alert_manager.reset(datetime.now().astimezone())
        self._apply_alert_update(update)
        icon.stop()

    def _make_icon(self, color: str) -> Image.Image:
        """Render a compact posture indicator for the notification area.

        Args:
            color: One of green, amber, or gray.

        Returns:
            A 64 by 64 Pillow image.
        """
        colors = {"green": "#28A879", "amber": "#E2A52B", "gray": "#92999B"}
        image = Image.open(resource_path("logo/logo.png")).convert("RGBA")
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.ellipse((43, 43, 63, 63), fill="white")
        draw.ellipse((46, 46, 60, 60), fill=colors[color])
        return image

    def _set_icon(self, color: str) -> None:
        """Update the visible tray icon color.

        Args:
            color: One of green, amber, or gray.

        Returns:
            None.
        """
        self._icon.icon = self._make_icon(color)

    def _show_message(self, message: str, error: bool = False) -> None:
        """Show a small native message dialog.

        Args:
            message: Text to show to the user.
            error: Whether to use an error dialog.

        Returns:
            None.
        """
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if error:
                messagebox.showerror(APP_NAME, message, parent=root)
            else:
                messagebox.showinfo(APP_NAME, message, parent=root)
        finally:
            root.destroy()

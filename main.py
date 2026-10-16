"""SpineUp offline posture monitor entry point."""

import ctypes
import msvcrt
import os
import sqlite3

if os.name == "nt":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("spineup.app")

from calibration import load_calibration
from config import CAMERA_INDEX, DATA_DIR, DATABASE_PATH, MODEL_PATH, resource_path
from database import PostureDatabase
from pose_detector import PoseDetector, probe_camera
from storage import SessionStorage
from ui import SpineUpWindow
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk


_instance_lock = None


def acquire_single_instance() -> bool:
    """Hold a Windows file lock so a second launch exits quietly."""
    global _instance_lock
    lock_file = None
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        lock_file = (DATA_DIR / "spineup.lock").open("a+b")
        lock_file.seek(0)
        if not lock_file.read(1):
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        if lock_file is not None:
            lock_file.close()
        return False
    _instance_lock = lock_file
    return True


def main() -> int:
    """Initialize local resources and run the SpineUp desktop window.

    Returns:
        Process exit status; zero indicates a normal shutdown.
    """
    if not acquire_single_instance():
        return 0
    print("[SpineUp] Запуск оффлайн-монитора осанки.")
    print(f"[SpineUp] Локальная модель: {MODEL_PATH}")
    try:
        detector = PoseDetector(MODEL_PATH)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[SpineUp] Не удалось загрузить модель: {error}")
        return 1
    print("[SpineUp] MediaPipe Pose Landmarker загружен (Tasks API).")

    camera_available, camera_message = probe_camera(CAMERA_INDEX)
    print(f"[SpineUp] Проверка камеры: {camera_message}")
    if not camera_available:
        print("[SpineUp] Приложение останется в трее; камеру можно проверить повторно после запуска.")

    calibration = load_calibration()
    print("[SpineUp] Статус калибровки: готова." if calibration else "[SpineUp] Статус калибровки: не выполнена.")
    database = PostureDatabase(DATABASE_PATH)
    database.close_stale_episodes()
    print(f"[SpineUp] История хранится локально: {DATABASE_PATH}")
    try:
        storage = SessionStorage()
        print(f"[SpineUp] Статистика хранится локально: {storage.database_path}")
    except (OSError, sqlite3.Error) as error:
        storage = None
        print(f"[SpineUp] Не удалось инициализировать статистику: {error}")
    application = SpineUpWindow(
        detector, database, calibration, CAMERA_INDEX, storage=storage
    )
    application.run()
    return 0
def show_gif(path, label):
    gif = Image.open(resource_path(path))

    frames = [
        ImageTk.PhotoImage(frame.copy().convert("RGBA"))
        for frame in ImageSequence.Iterator(gif)
    ]

    def animate(index=0):
        label.config(image=frames[index])
        label.image = frames[index]

        label.after(100, animate, (index + 1) % len(frames))

    animate()
    


if __name__ == "__main__":
    raise SystemExit(main())

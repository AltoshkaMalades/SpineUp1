"""SpineUp offline posture monitor entry point."""

import ctypes
import os

if os.name == "nt":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("spineup.app")

from calibration import load_calibration
from config import CAMERA_INDEX, DATABASE_PATH, MODEL_PATH
from database import PostureDatabase
from pose_detector import PoseDetector, probe_camera
from ui import SpineUpWindow
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk


def main() -> int:
    """Initialize local resources and run the SpineUp desktop window.

    Returns:
        Process exit status; zero indicates a normal shutdown.
    """
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
    application = SpineUpWindow(detector, database, calibration, CAMERA_INDEX)
    application.run()
    return 0
def show_gif(path, label):
    gif = Image.open(path)

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


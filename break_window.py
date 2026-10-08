"""Posture break screen: shows exercises one by one with a countdown."""

import tkinter as tk
from typing import Callable, Sequence

from PIL import Image, ImageSequence, ImageTk

from exercises import Exercise

GIF_DELAY_MS = 300  # чем больше число, тем медленнее гифка


class BreakWindow:
    """Window that walks the user through the exercises on a timer."""

    def __init__(
        self,
        parent: tk.Misc,
        exercises: Sequence[Exercise],
        on_finished: Callable[[bool], None],
    ) -> None:
        """Open the window and start the countdown.

        Args:
            parent: Main Tk window.
            exercises: Exercises to show in order.
            on_finished: Called once with True if the break was completed,
                False if the user stopped it early.
        """
        self.exercises = exercises
        self.on_finished = on_finished
        self.total_seconds = sum(item.seconds for item in exercises)
        self.remaining = self.total_seconds
        self._after_id: str | None = None
        self._done = False

        self._gif_frames = []
        self._gif_index = 0
        self._gif_path: str | None = None
        self._gif_after_id: str | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("Перерыв для осанки")
        self.window.geometry("520x360")
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        self.countdown_label = tk.Label(self.window, font=("Segoe UI", 40, "bold"), pady=8)
        self.countdown_label.pack()
        self.title_label = tk.Label(self.window, font=("Segoe UI", 16, "bold"))
        self.title_label.pack()
        self.description_label = tk.Label(
            self.window, font=("Segoe UI", 12), wraplength=460, justify="center", pady=12
        )
        self.description_label.pack()
        self.step_label = tk.Label(self.window, font=("Segoe UI", 10), fg="#4A5558")
        self.step_label.pack()
        self.animation_label = tk.Label(self.window)
        self.animation_label.pack(pady=10)

        tk.Button(
            self.window, text="Закончить раньше", command=self.cancel,
            font=("Segoe UI", 11), padx=10, pady=5,
        ).pack(pady=14)

        self._render()
        self._after_id = self.window.after(1000, self._tick)

    def _current_exercise(self) -> tuple[int, Exercise]:
        """Pick the exercise that matches the elapsed time."""
        elapsed = self.total_seconds - self.remaining
        boundary = 0
        for index, item in enumerate(self.exercises):
            boundary += item.seconds
            if elapsed < boundary:
                return index, item
        return len(self.exercises) - 1, self.exercises[-1]

    def _show_gif(self, path: str) -> None:
        """Load the GIF and start the animation (only if the path changed)."""
        if path == self._gif_path:
            return
        self._gif_path = path

        if self._gif_after_id is not None:
            self.window.after_cancel(self._gif_after_id)
            self._gif_after_id = None

        gif = Image.open(path)
        self._gif_frames = []
        for frame in ImageSequence.Iterator(gif):
            frame = frame.copy().convert("RGBA")
            frame.thumbnail((460, 220))
            self._gif_frames.append(ImageTk.PhotoImage(frame))

        self._gif_index = 0
        self._animate_gif()

    def _animate_gif(self) -> None:
        if not self._gif_frames:
            return
        self.animation_label.configure(image=self._gif_frames[self._gif_index])
        self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
        self._gif_after_id = self.window.after(GIF_DELAY_MS, self._animate_gif)

    def _render(self) -> None:
        index, item = self._current_exercise()
        self.countdown_label.configure(text=f"{self.remaining} с")
        self.title_label.configure(text=item.name)
        self.description_label.configure(text=item.description)
        self.step_label.configure(text=f"Упражнение {index + 1} из {len(self.exercises)}")
        self._show_gif(item.animation)

    def _tick(self) -> None:
        self.remaining -= 1
        if self.remaining <= 0:
            self._finish(True)
            return
        self._render()
        self._after_id = self.window.after(1000, self._tick)

    def _finish(self, completed: bool) -> None:
        if self._done:
            return
        self._done = True
        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
        if self._gif_after_id is not None:
            self.window.after_cancel(self._gif_after_id)
        self.window.destroy()
        self.on_finished(completed)

    def cancel(self) -> None:
        """Stop the break early (button or window close)."""
        self._finish(False)
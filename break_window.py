"""Posture break screen: shows exercises one by one with a countdown."""

import tkinter as tk
from typing import Callable, Sequence

from PIL import Image, ImageSequence, ImageTk

from config import BREAK_DURATION_SECONDS, THEME, resource_path
from exercises import Exercise
from window_chrome import create_title_bar, make_frameless

GIF_DELAY_MS = 300  # чем больше число, тем медленнее гифка
BACKGROUND = THEME["background"]
SURFACE = THEME["surface"]
TEXT = THEME["text"]
MUTED = THEME["muted"]
ACCENT = THEME["accent"]
FONT = THEME["font"]


def _load_gif_frames(path: str, size: tuple[int, int]) -> list[ImageTk.PhotoImage]:
    """Load and resize GIF frames for Tk windows."""
    gif = Image.open(resource_path(path))
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frame = frame.copy().convert("RGBA")
        frame.thumbnail(size)
        frames.append(ImageTk.PhotoImage(frame))
    return frames


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
        self.window.geometry("560x570")
        self.window.minsize(500, 520)
        self.window.configure(bg=THEME["border"])
        make_frameless(self.window)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        create_title_bar(
            self.window, "Перерыв для осанки", on_close=self.cancel
        ).pack(fill="x", padx=1, pady=(1, 0))
        content = tk.Frame(self.window, bg=BACKGROUND, padx=18, pady=10)
        content.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.countdown_label = tk.Label(
            content,
            font=(FONT, 40, "bold"),
            pady=8,
            bg=BACKGROUND,
            fg=ACCENT,
        )
        self.countdown_label.pack()
        self.title_label = tk.Label(
            content, font=(FONT, 16, "bold"), bg=BACKGROUND, fg=TEXT
        )
        self.title_label.pack()
        self.description_label = tk.Label(
            content,
            font=(FONT, 11),
            wraplength=460,
            justify="center",
            pady=12,
            bg=BACKGROUND,
            fg=TEXT,
        )
        self.description_label.pack()
        self.step_label = tk.Label(
            content, font=(FONT, 10), fg=MUTED, bg=BACKGROUND
        )
        self.step_label.pack()
        self.progress = tk.Canvas(
            content, height=10, bg=THEME["surface"], highlightthickness=0, bd=0
        )
        self.progress.pack(fill="x", pady=(5, 3))
        self.animation_label = tk.Label(content, bg=BACKGROUND)
        self.animation_label.pack(pady=10)

        tk.Button(
            content, text="Закончить раньше", command=self.cancel,
            font=(FONT, 10, "bold"),
            padx=10,
            pady=7,
            bg=SURFACE,
            fg=TEXT,
            activebackground=THEME["surface_hover"],
            activeforeground=TEXT,
            relief="flat",
            cursor="hand2",
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

        self._gif_frames = _load_gif_frames(path, (460, 220))

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
        self.progress.delete("all")
        width = max(self.progress.winfo_width(), 360)
        self.progress.create_rectangle(
            0,
            0,
            width * (self.total_seconds - self.remaining) / max(self.total_seconds, 1),
            10,
            fill=ACCENT,
            outline="",
        )
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


class BreakPopup:
    """Compact reminder popup for one exercise and a one-minute timer."""

    def __init__(
        self,
        parent: tk.Misc,
        exercise: Exercise,
        on_finished: Callable[[str], None],
    ) -> None:
        self.on_finished = on_finished
        self.remaining = BREAK_DURATION_SECONDS
        self._done = False
        self._gif_frames = _load_gif_frames(exercise.animation, (150, 105))
        self._gif_index = 0
        self._gif_after_id: str | None = None
        self._timer_after_id: str | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("SpineUp")
        self.window.resizable(False, False)
        self.window.configure(bg=THEME["border"])
        make_frameless(self.window)
        self.window.attributes("-topmost", True)
        self.window.focusmodel("passive")
        self.window.protocol("WM_DELETE_WINDOW", self.later)

        create_title_bar(
            self.window, "SpineUp", on_close=self.later, controls=False
        ).pack(fill="x", padx=1, pady=(1, 0))
        content = tk.Frame(self.window, bg=BACKGROUND)
        content.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        tk.Label(
            content,
            text="Пора размяться — сделайте минутный перерыв.",
            font=("Segoe UI", 11, "bold"),
            wraplength=320,
            justify="center",
            padx=14,
            pady=10,
            bg=BACKGROUND,
            fg=TEXT,
        ).pack(fill="x")
        tk.Label(
            content,
            text=exercise.name,
            font=("Segoe UI", 11),
            wraplength=320,
            justify="center",
            padx=12,
            bg=BACKGROUND,
            fg=MUTED,
        ).pack(fill="x")
        self.animation_label = tk.Label(content, bg=BACKGROUND)
        self.animation_label.pack(pady=(4, 0))
        self.countdown_label = tk.Label(
            content,
            text="60 с",
            font=(FONT, 18, "bold"),
            bg=BACKGROUND,
            fg=ACCENT,
        )
        self.countdown_label.pack(pady=4)

        buttons = tk.Frame(content, bg=BACKGROUND)
        buttons.pack(pady=(0, 10))
        button_style = {
            "bg": SURFACE,
            "fg": TEXT,
            "activebackground": THEME["surface_hover"],
            "activeforeground": TEXT,
            "relief": "flat",
            "cursor": "hand2",
            "font": (FONT, 9, "bold"),
            "pady": 5,
        }
        tk.Button(buttons, text="Готово", command=self.done, width=10, **button_style).pack(
            side="left", padx=6
        )
        tk.Button(buttons, text="Позже", command=self.later, width=10, **button_style).pack(
            side="left", padx=6
        )

        self.window.update_idletasks()
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        x = self.window.winfo_screenwidth() - width - 16
        y = self.window.winfo_screenheight() - height - 48
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self._animate_gif()
        self._timer_after_id = self.window.after(1000, self._tick)

    def _animate_gif(self) -> None:
        if not self._gif_frames or self._done:
            return
        frame = self._gif_frames[self._gif_index]
        self.animation_label.configure(image=frame)
        self.animation_label.image = frame
        self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
        self._gif_after_id = self.window.after(GIF_DELAY_MS, self._animate_gif)

    def _tick(self) -> None:
        self.remaining -= 1
        if self.remaining <= 0:
            self._finish("done")
            return
        self.countdown_label.configure(text=f"{self.remaining} с")
        self._timer_after_id = self.window.after(1000, self._tick)

    def done(self) -> None:
        self._finish("done")

    def later(self) -> None:
        self._finish("later")

    def _finish(self, action: str) -> None:
        if self._done:
            return
        self._done = True
        if self._timer_after_id is not None:
            self.window.after_cancel(self._timer_after_id)
        if self._gif_after_id is not None:
            self.window.after_cancel(self._gif_after_id)
        self.window.destroy()
        self.on_finished(action)
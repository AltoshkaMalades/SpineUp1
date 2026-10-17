"""Tkinter dashboard for local SpineUp session statistics."""

import logging
import sqlite3
import tkinter as tk
from datetime import datetime
from typing import Callable

from config import THEME
from storage import SessionStorage
from window_chrome import create_title_bar, make_frameless, show_themed_message


logger = logging.getLogger(__name__)
BACKGROUND = THEME["background"]
SURFACE = THEME["surface"]
SURFACE_RAISED = THEME["surface_hover"]
TEXT = THEME["text"]
MUTED = THEME["muted"]
ACCENT = THEME["accent"]
GRID = THEME["border"]
FONT = THEME["font"]
CHART_WARNING_THRESHOLD_PERCENT = 15.0


class StatisticsWindow:
    """Display a seven-day chart, weekly cards, and recent sessions."""

    def __init__(
        self,
        parent: tk.Misc,
        storage: SessionStorage | None,
        on_load_demo: Callable[[], int],
    ) -> None:
        self.storage = storage
        self.on_load_demo = on_load_demo
        self.window = tk.Toplevel(parent)
        self.window.title("SpineUp · Statistics")
        self.window.geometry("920x720")
        self.window.minsize(760, 600)
        self.window.configure(bg=GRID)
        make_frameless(self.window)
        self.window.transient(parent)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        create_title_bar(
            self.window,
            "SpineUp · Статистика",
            on_close=self.window.destroy,
        ).grid(row=0, column=0, sticky="ew", padx=1, pady=(1, 0))

        content = tk.Frame(self.window, bg=BACKGROUND, padx=22, pady=18)
        content.grid(row=1, column=0, sticky="nsew", padx=1, pady=(0, 1))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1)

        heading = tk.Frame(content, bg=BACKGROUND)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        heading.columnconfigure(0, weight=1)
        tk.Label(
            heading,
            text="Статистика",
            bg=BACKGROUND,
            fg=TEXT,
            font=(FONT, 24, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            heading,
            text="Локальная история осанки · последние 7 дней",
            bg=BACKGROUND,
            fg=MUTED,
            font=(FONT, 10),
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        cards = tk.Frame(content, bg=BACKGROUND)
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        for column in range(4):
            cards.columnconfigure(column, weight=1, uniform="stat-card")
        self.card_values: list[tk.Label] = []
        for column, title in enumerate(
            ("Сутулость за неделю", "Изменение к прошлой", "Перерывы", "Серия дней")
        ):
            card = tk.Frame(
                cards,
                bg=SURFACE,
                highlightbackground=GRID,
                highlightthickness=1,
                padx=14,
                pady=13,
            )
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 10 if column < 3 else 0))
            tk.Label(
                card,
                text=title,
                bg=SURFACE,
                fg=MUTED,
                font=(FONT, 9),
                anchor="w",
                wraplength=155,
            ).pack(fill="x")
            value = tk.Label(
                card,
                text="—",
                bg=SURFACE,
                fg=TEXT,
                font=(FONT, 21, "bold"),
                anchor="w",
                pady=5,
            )
            value.pack(fill="x")
            self.card_values.append(value)

        chart_panel = tk.Frame(
            content,
            bg=SURFACE,
            highlightbackground=GRID,
            highlightthickness=1,
            padx=16,
            pady=12,
        )
        chart_panel.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        chart_panel.columnconfigure(0, weight=1)
        tk.Label(
            chart_panel,
            text="Доля кадров с сутулостью",
            bg=SURFACE,
            fg=TEXT,
            font=(FONT, 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.chart = tk.Canvas(
            chart_panel,
            height=190,
            bg=SURFACE,
            highlightthickness=0,
            bd=0,
        )
        self.chart.grid(row=1, column=0, sticky="ew")
        self.chart.bind("<Configure>", self._draw_chart)

        sessions_panel = tk.Frame(content, bg=BACKGROUND)
        sessions_panel.grid(row=3, column=0, sticky="nsew")
        sessions_panel.columnconfigure(0, weight=1)
        sessions_panel.rowconfigure(1, weight=1)
        tk.Label(
            sessions_panel,
            text="Недавние сессии",
            bg=BACKGROUND,
            fg=TEXT,
            font=(FONT, 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.sessions_list = tk.Listbox(
            sessions_panel,
            height=8,
            bg=SURFACE,
            fg=TEXT,
            selectbackground=SURFACE_RAISED,
            selectforeground=TEXT,
            font=(FONT, 10),
            relief="flat",
            highlightthickness=1,
            highlightbackground=GRID,
            activestyle="none",
            borderwidth=0,
        )
        self.sessions_list.grid(row=1, column=0, sticky="nsew")

        footer = tk.Frame(content, bg=BACKGROUND)
        footer.grid(row=4, column=0, sticky="e", pady=(12, 0))
        self.data_status = tk.Label(
            footer, text="", bg=BACKGROUND, fg=MUTED, font=(FONT, 9)
        )
        self.data_status.pack(side="left", padx=(0, 12))
        tk.Button(
            footer,
            text="Загрузить демо-данные",
            command=self._load_demo,
            bg=SURFACE_RAISED,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground=BACKGROUND,
            relief="flat",
            padx=12,
            pady=7,
            font=(FONT, 9, "bold"),
            cursor="hand2",
        ).pack(side="right")
        self._daily_stats: list[dict[str, object]] = []
        self.refresh()

    def refresh(self) -> None:
        """Reload dashboard values from the local SQLite store."""
        if self.storage is None:
            self.data_status.configure(text="Хранилище недоступно")
            return
        try:
            self._daily_stats = self.storage.get_daily_stats(7)
            summary = self.storage.get_weekly_summary()
            recent = self.storage.get_recent_sessions(10)
        except (sqlite3.Error, OSError) as error:
            logger.exception("Could not load SpineUp statistics")
            self.data_status.configure(text="Не удалось загрузить статистику")
            show_themed_message(
                self.window,
                "SpineUp",
                f"Не удалось прочитать локальную статистику:\n{error}",
                "error",
            )
            return

        change = float(summary["slouch_change_percent_points"])
        change_text = f"{change:+.1f} п.п."
        self.card_values[0].configure(
            text=f"{summary['slouch_percent_this_week']:.1f}%"
        )
        self.card_values[1].configure(
            text=change_text, fg=THEME["good_text"] if change <= 0 else THEME["coral_text"]
        )
        self.card_values[2].configure(text=str(summary["completed_breaks"]))
        self.card_values[3].configure(text=f"{summary['streak_days']} дн.")
        self.sessions_list.delete(0, tk.END)
        if not recent:
            self.sessions_list.insert(tk.END, "Сессий пока нет — начните мониторинг.")
        for session in recent:
            started_at = datetime.fromisoformat(str(session["started_at"])).astimezone()
            duration = session["duration_sec"]
            duration_label = (
                f"{float(duration) / 60:.0f} мин" if duration is not None else "идёт"
            )
            slouch_percent = session["slouch_percent"]
            percent_label = (
                f"{float(slouch_percent):.1f}% сутулости"
                if slouch_percent is not None
                else "нет кадров"
            )
            self.sessions_list.insert(
                tk.END,
                f"{started_at:%d.%m.%Y  %H:%M}   ·   {duration_label}   ·   {percent_label}",
            )
        self.data_status.configure(text="Данные остаются на этом устройстве")
        self._draw_chart()

    def _load_demo(self) -> None:
        try:
            added = self.on_load_demo()
        except (sqlite3.Error, OSError, RuntimeError, ValueError) as error:
            logger.exception("Could not load SpineUp demo data")
            show_themed_message(
                self.window,
                "SpineUp",
                f"Не удалось загрузить демо-данные:\n{error}",
                "error",
            )
            return
        self.data_status.configure(text=f"Добавлено демо-сессий: {added}")
        self.refresh()

    def _draw_chart(self, _event: object | None = None) -> None:
        del _event
        if not hasattr(self, "chart"):
            return
        canvas = self.chart
        canvas.delete("all")
        width = max(canvas.winfo_width(), 360)
        height = max(canvas.winfo_height(), 170)
        left, right, top, bottom = 40, width - 12, 12, height - 30
        chart_height = bottom - top
        for percent in (0, 25, 50, 75, 100):
            y = bottom - chart_height * percent / 100
            canvas.create_line(left, y, right, y, fill=GRID, width=1)
            canvas.create_text(
                left - 8,
                y,
                text=str(percent),
                fill=MUTED,
                font=(FONT, 8),
                anchor="e",
            )
        if not self._daily_stats:
            return
        slot_width = (right - left) / len(self._daily_stats)
        bar_width = min(34, slot_width * 0.58)
        for index, item in enumerate(self._daily_stats):
            x_center = left + slot_width * (index + 0.5)
            percent = item["slouch_percent"]
            if percent is not None:
                bar_height = chart_height * float(percent) / 100.0
                canvas.create_rectangle(
                    x_center - bar_width / 2,
                    bottom - bar_height,
                    x_center + bar_width / 2,
                    bottom,
                    fill=(
                        THEME["coral_text"]
                        if float(percent) >= CHART_WARNING_THRESHOLD_PERCENT
                        else ACCENT
                    ),
                    outline="",
                )
            day = item["date"]
            canvas.create_text(
                x_center,
                bottom + 15,
                text=day.strftime("%d.%m"),
                fill=MUTED,
                font=(FONT, 8),
            )

"""Shared light-themed chrome for SpineUp Tk windows."""

import ctypes
import os
import tkinter as tk
from ctypes import wintypes
from typing import Callable

from config import THEME


def apply_taskbar_style(window: tk.Misc) -> None:
    """Mark a frameless Windows top-level as an application window."""
    if os.name != "nt":
        return
    try:
        hwnd = int(window.wm_frame(), 16)
    except (AttributeError, tk.TclError, ValueError):
        hwnd = window.winfo_id()
    user32 = ctypes.windll.user32
    get_window_long = user32.GetWindowLongW
    get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
    get_window_long.restype = ctypes.c_long
    set_window_long = user32.SetWindowLongW
    set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    set_window_long.restype = ctypes.c_long
    set_window_pos = user32.SetWindowPos
    set_window_pos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    set_window_pos.restype = wintypes.BOOL
    ex_style = get_window_long(hwnd, -20)
    ex_style = (ex_style | 0x00040000) & ~0x00000080
    set_window_long(hwnd, -20, ex_style)
    set_window_pos(hwnd, None, 0, 0, 0, 0, 0x0027)


def make_frameless(window: tk.Toplevel | tk.Tk, taskbar: bool = False) -> None:
    """Remove the native frame and preserve normal taskbar behavior if requested."""
    window.overrideredirect(True)
    if not taskbar:
        return

    def restore_frame_state(_event: tk.Event | None = None) -> None:
        if not window.winfo_exists():
            return
        window.after_idle(lambda: _finish_restore(window))

    window.bind("<Map>", restore_frame_state, add="+")
    window.after_idle(lambda: apply_taskbar_style(window))


def _finish_restore(window: tk.Misc) -> None:
    if not window.winfo_exists():
        return
    window.overrideredirect(True)
    apply_taskbar_style(window)


def create_title_bar(
    window: tk.Toplevel | tk.Tk,
    title: str,
    icon: tk.PhotoImage | None = None,
    on_close: Callable[[], None] | None = None,
    controls: bool = True,
) -> tk.Frame:
    """Create a draggable custom title bar with minimize/maximize/close controls."""
    bar = tk.Frame(window, bg=THEME["surface"], height=38)
    bar.pack_propagate(False)
    bar.columnconfigure(1, weight=1)
    if icon is not None:
        tk.Label(bar, image=icon, bg=THEME["surface"]).grid(
            row=0, column=0, padx=(12, 7), pady=5
        )
    tk.Label(
        bar,
        text=title,
        bg=THEME["surface"],
        fg=THEME["text"],
        font=(THEME["font"], 10, "bold"),
        anchor="w",
    ).grid(row=0, column=1, sticky="ew", pady=5)

    state: dict[str, str | bool] = {
        "maximized": False,
        "geometry": window.geometry(),
    }
    maximize_button: tk.Button | None = None

    def toggle_maximize(_event: tk.Event | None = None) -> None:
        if state["maximized"]:
            window.geometry(str(state["geometry"]))
            state["maximized"] = False
            if maximize_button is not None:
                maximize_button.configure(text="□")
            return
        state["geometry"] = window.geometry()
        if os.name == "nt":
            try:
                class Rect(ctypes.Structure):
                    _fields_ = [
                        ("left", ctypes.c_long),
                        ("top", ctypes.c_long),
                        ("right", ctypes.c_long),
                        ("bottom", ctypes.c_long),
                    ]

                rect = Rect()
                system_parameters_info = ctypes.windll.user32.SystemParametersInfoW
                system_parameters_info.argtypes = [
                    wintypes.UINT,
                    wintypes.UINT,
                    wintypes.LPVOID,
                    wintypes.UINT,
                ]
                system_parameters_info.restype = wintypes.BOOL
                succeeded = system_parameters_info(
                    0x0030, 0, ctypes.byref(rect), 0
                )
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                x, y = rect.left, rect.top
                if not succeeded or width <= 0 or height <= 0:
                    raise OSError("Unable to read the Windows work area")
            except (AttributeError, OSError):
                width, height, x, y = (
                    window.winfo_screenwidth(),
                    window.winfo_screenheight(),
                    0,
                    0,
                )
        else:
            width, height, x, y = (
                window.winfo_screenwidth(),
                window.winfo_screenheight(),
                0,
                0,
            )
        window.geometry(f"{width}x{height}+{x}+{y}")
        state["maximized"] = True
        if maximize_button is not None:
            maximize_button.configure(text="❐")

    if controls:
        controls_frame = tk.Frame(bar, bg=THEME["surface"])
        controls_frame.grid(row=0, column=2, sticky="e")

        def title_button(
            label: str,
            command: Callable[[], None],
            hover: str = THEME["surface_hover"],
        ) -> tk.Button:
            button = tk.Button(
                controls_frame,
                text=label,
                command=command,
                bg=THEME["surface"],
                fg=THEME["text"],
                activebackground=hover,
                activeforeground=THEME["text"],
                relief="flat",
                borderwidth=0,
                font=(THEME["font"], 11),
                width=4,
                cursor="hand2",
            )
            button.pack(side="left", fill="y")
            button.bind("<Enter>", lambda _event: button.configure(bg=hover))
            button.bind("<Leave>", lambda _event: button.configure(bg=THEME["surface"]))
            return button

        def minimize() -> None:
            window.iconify()

        title_button("—", minimize)
        maximize_button = title_button("□", toggle_maximize)
        close_button = title_button("×", on_close or window.destroy, THEME["coral"])
        close_button.bind("<Enter>", lambda _event: close_button.configure(
            bg=THEME["coral_text"], fg=THEME["background"]
        ))
        close_button.bind("<Leave>", lambda _event: close_button.configure(
            bg=THEME["surface"], fg=THEME["text"]
        ))

    drag = {"x": 0, "y": 0, "left": 0, "top": 0}

    def begin_drag(event: tk.Event) -> None:
        if state["maximized"]:
            return
        drag["x"], drag["y"] = event.x_root, event.y_root
        drag["left"], drag["top"] = window.winfo_x(), window.winfo_y()

    def move_drag(event: tk.Event) -> None:
        if state["maximized"]:
            return
        x = drag["left"] + event.x_root - drag["x"]
        y = drag["top"] + event.y_root - drag["y"]
        window.geometry(f"+{x}+{y}")

    for widget in (bar, *bar.winfo_children()):
        widget.bind("<ButtonPress-1>", begin_drag, add="+")
        widget.bind("<B1-Motion>", move_drag, add="+")
        if controls:
            widget.bind("<Double-Button-1>", toggle_maximize, add="+")
    return bar


def show_themed_message(
    parent: tk.Misc,
    title: str,
    message: str,
    kind: str = "info",
) -> None:
    """Show a synchronous, themed message dialog."""
    dialog = tk.Toplevel(parent)
    dialog.overrideredirect(True)
    dialog.configure(bg=THEME["border"])
    dialog.transient(parent)
    dialog.grab_set()
    header = create_title_bar(dialog, title, controls=False)
    header.pack(fill="x", padx=1, pady=(1, 0))
    body = tk.Frame(dialog, bg=THEME["background"], padx=20, pady=16)
    body.pack(fill="both", expand=True, padx=1, pady=(0, 1))
    color = THEME["coral_text"] if kind == "error" else THEME["text"]
    tk.Label(
        body,
        text=message,
        bg=THEME["background"],
        fg=color,
        font=(THEME["font"], 10),
        justify="left",
        wraplength=420,
    ).pack(fill="x", pady=(0, 16))
    tk.Button(
        body,
        text="OK",
        command=dialog.destroy,
        bg=THEME["accent"],
        fg=THEME["background"],
        activebackground=THEME["accent_hover"],
        activeforeground=THEME["background"],
        relief="flat",
        padx=18,
        pady=6,
        font=(THEME["font"], 10, "bold"),
        cursor="hand2",
    ).pack(anchor="e")
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.update_idletasks()
    width, height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    x = parent.winfo_rootx() + max((parent.winfo_width() - width) // 2, 0)
    y = parent.winfo_rooty() + max((parent.winfo_height() - height) // 2, 0)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    dialog.wait_window()

"""Windows startup registration for SpineUp."""

import sys
import winreg

from config import APP_NAME, PROJECT_DIR


RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _command() -> str:
    """Build the Run-key command for frozen and source executions."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" "{PROJECT_DIR / "main.py"}"'


def enable_autostart() -> None:
    """Register SpineUp to start when the current Windows user signs in."""
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, _command())


def disable_autostart() -> None:
    """Remove SpineUp's current-user startup registration if present."""
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        pass


def is_autostart_enabled() -> bool:
    """Return whether SpineUp has a current-user startup registration."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_NAME)
        return True
    except FileNotFoundError:
        return False
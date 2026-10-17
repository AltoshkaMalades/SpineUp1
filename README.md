# SpineUp

SpineUp is a privacy-first Windows desktop posture companion. It uses a webcam to notice sustained slouching and gently remind you to reset your posture. Long periods at a desk can make it easy to miss fatigue and gradual posture changes; SpineUp adds lightweight, local feedback without uploading camera footage or posture history.

## Features

- Live posture tracking with a personal camera calibration.
- Local pose analysis with MediaPipe and OpenCV; frames are not uploaded.
- Gentle posture alerts, a system-tray menu, and a one-minute exercise break with GIF demonstrations.
- Local session history, slouch-rate charts, weekly comparisons, completed-break counts, and streaks.
- Optional demo data for presenting a populated statistics dashboard.
- Optional launch-at-sign-in setting.

## Screenshots

Replace these placeholders with project screenshots before publishing:

![Main window screenshot placeholder](screenshots/main-window.png)

![Statistics dashboard screenshot placeholder](screenshots/statistics.png)

![Exercise break screenshot placeholder](screenshots/exercise-break.png)

## Run from source

Requirements: Windows, Python 3.10 or newer, and a webcam.

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Allow camera access when Windows requests it. Complete the in-app camera calibration before monitoring starts. A webcam and the included `models/pose_landmarker_lite.task` model are required for live posture detection.

Run the unit tests with:

```powershell
python -m unittest discover -s tests
```

## Download the Windows application

Download the latest packaged SpineUp build from [GitHub Releases](https://github.com/AltoshkaMalades/SpineUp1/releases) (release download placeholder). Extract the archive and run `SpineUp.exe`.

To build the Windows application locally:

```powershell
python -m pip install -r requirements-build.txt
.\build.bat
```

The executable is created at `dist\SpineUp\SpineUp.exe`. PyInstaller collects the Python modules imported by the app and bundles the local model, icon, and exercise GIFs through `SpineUp.spec`.

## Privacy and local data

Pose inference and posture calculations run on the computer. SpineUp does not send webcam frames, calibration data, or statistics to a server.

The statistics database is created automatically at `%APPDATA%\SpineUp\spineup.db`. It contains monitoring sessions (start/end time, duration, total frames, and slouch frames) and timestamped posture/break events. `storage.py` batches frame counts in memory and writes them to SQLite in groups rather than writing every camera sample.

The existing calibration file and episode history remain in `%USERPROFILE%\.spineup`; the new session database is separate so existing app data is preserved. The calculations in `stats.py` aggregate frame-weighted daily and weekly slouch percentages, completed breaks, and consecutive days below 25% slouching. No third-party database server or new runtime dependency is required.

## Tech stack

- Python and Tkinter for the Windows desktop interface.
- SQLite (`sqlite3`) for local session and event storage.
- MediaPipe Pose and OpenCV for on-device camera analysis.
- Pillow and GIF assets for the exercise demonstrations.
- pystray for the Windows system-tray menu.
- PyInstaller for packaging.

## Architecture

`main.py` initializes the local pose detector, the existing posture-episode database, the new session store, and the Tkinter application. The camera preview runs on Tk's event loop; pose inference runs on a worker thread and passes results back to the UI. `AlertManager` handles debounced posture transitions. The UI records valid sampled frames and session events through `SessionStorage` in `storage.py`; `stats.py` keeps the date/weekly calculations independent of SQLite. The Statistics window queries this local API and draws its chart with Tkinter's Canvas.

## Team and roles

| Name | Role |
| --- | --- |
| `Altynbek` | `[Project lead / product]` |
| `Altynbek` | `[Computer vision / posture analysis]` |
| `Altynbek` | `[Desktop UI / user experience]` |
| `Inkar` | `[Storage / testing / packaging]` |

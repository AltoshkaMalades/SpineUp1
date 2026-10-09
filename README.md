# SpineUp

SpineUp is a desktop posture monitor that uses your webcam and MediaPipe Pose to detect slouching and gently remind you to sit upright. Camera frames and posture analysis are processed locally; calibration and posture history are stored on your computer.

## Requirements

- Windows
- Python 3.10 or newer
- A webcam

The MediaPipe model is included at `models/pose_landmarker_lite.task`. The application uses OpenCV for camera capture, MediaPipe Pose for landmark detection, and Plyer for desktop notifications. Its other Python dependencies are listed in `requirements.txt`.

## Download and install

Download the latest `SpineUp` build from [GitHub Releases](https://github.com/AltoshkaMalades/SpineUp1/releases). Extract the downloaded archive, then run `SpineUp.exe`. On first run, complete the in-app posture calibration before starting monitoring. Open the tray menu and select **Launch at startup** to start SpineUp automatically when you sign in to Windows.

To build the Windows app yourself, install the build dependency and run the build script from the project directory:

```powershell
python -m pip install -r requirements-build.txt
.\build.bat
```

The packaged application is created in `dist\SpineUp\SpineUp.exe`.

## Run from source

In PowerShell, from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Allow camera access when Windows requests it. On first run, complete the in-app posture calibration before starting monitoring.

## Local data

Calibration and posture history are stored locally in `%USERPROFILE%\.spineup` (the user's home directory). They are not part of the project repository.

## Tests

```powershell
python -m unittest discover -s tests
```
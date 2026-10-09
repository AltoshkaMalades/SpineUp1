from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files


project_dir = Path(SPECPATH)
datas = [
    (str(project_dir / "logo" / "logo.png"), "logo"),
    (str(project_dir / "logo" / "logo.ico"), "logo"),
    (str(project_dir / "models" / "pose_landmarker_lite.task"), "models"),
]
datas.extend(
    (str(asset_path), "assets")
    for asset_path in (project_dir / "assets").glob("*.gif")
)
binaries = []
hiddenimports = []

for package in ("mediapipe", "cv2", "pystray", "plyer"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

datas += collect_data_files("mediapipe")
datas = list(dict.fromkeys(datas))

analysis = Analysis(
    [str(project_dir / "main.py")],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SpineUp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_dir / "logo" / "logo.ico")],
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SpineUp",
)
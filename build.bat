@echo off
setlocal
cd /d "%~dp0"

python -m PyInstaller --noconfirm --clean --distpath dist --workpath build SpineUp.spec
if errorlevel 1 exit /b 1

echo Build complete: dist\SpineUp\SpineUp.exe
exit /b 0
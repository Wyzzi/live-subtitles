@echo off
REM Double-click this (do not run it with "python list_devices.bat").
REM Lists speaker/headphone and microphone device names for subtitles.py.
cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\python.exe list_devices.py
echo.
pause

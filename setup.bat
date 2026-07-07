@echo off
REM One-time setup: creates a virtual environment and installs dependencies.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" during install, then run this script again.
    pause
    exit /b 1
)

if exist venv (
    venv\Scripts\python.exe -c "" >nul 2>nul
    if errorlevel 1 (
        echo Found a broken virtual environment ^(this usually happens if the
        echo project folder was copied from another computer^). Rebuilding it...
        rmdir /s /q venv
    )
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing dependencies (this can take a few minutes)...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo Setup complete. Next steps:
echo   1. Run list_devices.bat to find the exact device name.
echo   2. Edit run_subtitles.bat and set DEVICE to that name.
echo   3. Double-click run_subtitles.bat to start.
pause

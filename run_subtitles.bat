@echo off
REM Double-click this to start live subtitles.
REM First time: set DEVICE below to the exact name shown by list_devices.bat
REM (e.g. DEVICE=Headset Earphone (G435 Wireless Gaming Headset)).

setlocal
set DEVICE=
set MODEL=small
REM Set to 1 to caption a speaker/headphone device via loopback, so she
REM keeps hearing audio normally (this is what you want for Discord/video).
REM Set to 0 to record a microphone by name instead.
set LOOPBACK=1

cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if "%DEVICE%"=="" (
    echo DEVICE is not set yet.
    echo.
    echo Run list_devices.bat first to find the exact device name,
    echo then either:
    echo   - edit run_subtitles.bat and set DEVICE=that name, or
    echo   - enter it here just for this run.
    echo.
    set /p DEVICE=Enter device name now:
)

set EXTRA_ARGS=
if "%LOOPBACK%"=="1" set EXTRA_ARGS=--loopback

echo Starting live subtitles on device "%DEVICE%" with model "%MODEL%" (loopback=%LOOPBACK%)...
echo (Press Esc in the subtitle bar, or close this window, to stop.)
echo.

venv\Scripts\python.exe subtitles.py --device "%DEVICE%" --model %MODEL% %EXTRA_ARGS%

pause

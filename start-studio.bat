@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m video_ai.web
) else (
    python -m video_ai.web
)
if errorlevel 1 pause

@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HOME=%LOCALAPPDATA%\KoreanHumanizer\model-cache
if not exist ".venv-inference\Scripts\python.exe" (
  echo Please run Install.cmd first.
  pause
  exit /b 1
)
".venv-inference\Scripts\python.exe" scripts\run_desktop.py
if errorlevel 1 pause
endlocal

@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\install.ps1"
if errorlevel 1 (
  echo Installation stopped. See the message above.
  pause
  exit /b 1
)
endlocal

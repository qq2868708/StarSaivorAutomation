@echo off
setlocal
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0launch_ui.py" %*
set "launchExit=%errorlevel%"
if not "%launchExit%"=="0" pause
exit /b %launchExit%

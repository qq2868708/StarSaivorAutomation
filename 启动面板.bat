@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
if exist "%~dp0.venv\Scripts\python.exe" goto local
if defined STARSAVIOR_PYTHON goto custom
py -3.12 -c "pass" >nul 2>nul
if not errorlevel 1 goto python312
py -3.11 -c "pass" >nul 2>nul
if not errorlevel 1 goto python311
python "%~dp0tools\bootstrap_ui.py" %*
goto finish

:local
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\bootstrap_ui.py" %*
goto finish

:custom
"%STARSAVIOR_PYTHON%" "%~dp0tools\bootstrap_ui.py" %*
goto finish

:python312
py -3.12 "%~dp0tools\bootstrap_ui.py" %*
goto finish

:python311
py -3.11 "%~dp0tools\bootstrap_ui.py" %*

:finish
set "launchExit=%errorlevel%"
if not "%launchExit%"=="0" (
    echo Setup or startup failed. Details: runtime/bootstrap.log
    pause
)
exit /b %launchExit%

@echo off
setlocal
echo.
echo  DOL Analyser ^— one-time setup
echo  ================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Please install Python 3.10 or later from https://python.org
    echo  Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Create virtual environment (only on first run)
if not exist ".venv" (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Could not create virtual environment.
        pause & exit /b 1
    )
)

:: Activate and install dependencies
echo  Installing dependencies ^(this may take a minute^)...
call .venv\Scripts\activate.bat
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo.
echo  Setup complete!
echo  Run  run.bat  to launch the app.
echo.
pause

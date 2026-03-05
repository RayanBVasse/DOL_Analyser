@echo off
:: Always run from the folder that contains this script
cd /d "%~dp0"

if not exist ".venv" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Please run install.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo  ERROR: Could not activate virtual environment.
    echo.
    pause
    exit /b 1
)

echo  Starting DOL Analyser...
echo  Your browser will open at http://localhost:8501
echo  Close this window to stop the app.
echo.
streamlit run app.py --server.fileWatcherType=none --server.port=8501

pause

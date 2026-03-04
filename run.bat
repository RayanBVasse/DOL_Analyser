@echo off
if not exist ".venv" (
    echo  Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py

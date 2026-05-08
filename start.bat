@echo off
setlocal
echo ========================================================
echo KisanAI - Startup Script
echo ========================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip within the virtual environment
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Installing requirements...
pip install -r requirements.txt

REM Train ML models if they don't exist
if not exist backend\models\crop_model.pkl (
    echo [INFO] Training ML models - this will take a moment...
    python ml_training\train_models.py
) else (
    echo [INFO] ML models found. Skipping training.
)

REM Ensure .env exists
if not exist .env (
    echo [INFO] Creating .env file from .env.example...
    copy .env.example .env
    echo [WARNING] Please make sure to add your GROQ_API_KEY to the .env file!
)

REM Start FastAPI Server
echo [INFO] Starting FastAPI server on http://localhost:8000
echo ========================================================
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

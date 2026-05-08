#!/bin/bash
echo "========================================================"
echo "KisanAI - Startup Script"
echo "========================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# Train ML models if they don't exist
if [ ! -f "backend/models/crop_model.pkl" ]; then
    echo "[INFO] Training ML models (this will take a moment)..."
    python ml_training/train_models.py
else
    echo "[INFO] ML models found. Skipping training."
fi

# Ensure .env exists
if [ ! -f ".env" ]; then
    echo "[INFO] Creating .env file from .env.example..."
    cp .env.example .env
    echo "[WARNING] Please make sure to add your GROQ_API_KEY to the .env file!"
fi

# Start FastAPI Server
echo "[INFO] Starting FastAPI server on http://localhost:8000"
echo "========================================================"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

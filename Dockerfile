FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the ML training script to generate .pkl files
RUN python ml_training/train_models.py

# Expose the port the app runs on
EXPOSE 8000

# Start the application using Gunicorn with Uvicorn workers
CMD ["gunicorn", "backend.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

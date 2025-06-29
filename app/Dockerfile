# Dockerfile for Backend (FastAPI)
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run model download script
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py && rm /tmp/download_models.py

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
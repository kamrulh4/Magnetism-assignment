# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and PyTorch/YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights to package them inside the image
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy the rest of the application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the application using uvicorn
CMD ["python3", "main.py"]

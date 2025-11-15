# Base image: use official slim Python. For GPU/TensorFlow you'd normally pick an NVIDIA base image.
FROM python:3.10-slim

# Metadata
LABEL maintainer="Simerjeet Tech <simerjeet@simarjeettech.example>"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy only dependency files first (for Docker layer caching)
COPY requirements.txt /app/requirements.txt
COPY .dockerignore /app/.dockerignore

# Install pip deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create models and outputs folders (will be overridden if mounted)
RUN mkdir -p /app/models /app/data /app/outputs

# Expose Streamlit port
EXPOSE 8501

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Default command: run Streamlit app
CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.headless=true"]

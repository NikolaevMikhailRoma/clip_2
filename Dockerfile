# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install packages individually
RUN pip install --no-cache-dir -v torch
RUN pip install --no-cache-dir -v torchvision
RUN pip install --no-cache-dir -v transformers
RUN pip install --no-cache-dir -v pillow
RUN pip install --no-cache-dir -v numpy
RUN pip install --no-cache-dir -v tqdm
RUN pip install --no-cache-dir -v fastapi
RUN pip install --no-cache-dir -v uvicorn
RUN pip install --no-cache-dir -v requests
RUN pip install --no-cache-dir -v python-multipart

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]
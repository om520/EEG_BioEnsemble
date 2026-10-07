FROM python:3.11-slim

# Install minimal system dependencies required by mne/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire pipeline code into the container
COPY . .

# Ensure data directory exists inside the container
RUN mkdir -p /app/data

# Command to run the pipeline
CMD ["python", "main.py"]

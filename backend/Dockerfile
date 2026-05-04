# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Update apt and install the required OpenGL and core libraries for MediaPipe / OpenCV
RUN apt-get update && apt-get install -y \
    libgles2-mesa \
    libegl1-mesa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

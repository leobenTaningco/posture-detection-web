# Use a stable Debian Bookworm Python base image
FROM python:3.11-slim-bookworm

# Set Render environment variables for Python and Port
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set the working directory
WORKDIR /app

# Update apt and install correct dependencies for Debian Bookworm
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 10000

# Start the Flask app using Gunicorn on the correct port
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]

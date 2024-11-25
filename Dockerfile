# Use TensorFlow's official base image
#FROM tensorflow/tensorflow:latest
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app


# Upgrade pip and limit parallel threads
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_MAX_THREADS=1

# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies, pip uses thread to show progress bar. Try to disable it:
RUN pip install --progress-bar off -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

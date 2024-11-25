# Start with the TensorFlow image (CPU or GPU version)
FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install additional dependencies

RUN pip install --no-cache-dir -r requirements.txt --no-use-pep517

# Expose the application port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

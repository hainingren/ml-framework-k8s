import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from data.loader import DataLoader
import tensorflow as tf


tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
# Define input data schema using Pydantic
class PredictionRequest(BaseModel):
    ids: List[int]  # List of IDs for which predictions are needed

# Initialize FastAPI app
app = FastAPI()

# Paths to artifacts and data files
MODEL_PATH = "./artifacts/tf_model.h5"
CUSTOMERS_FILE = "./data/customers.csv"
NONCUSTOMERS_FILE = "./data/noncustomers.csv"
ACTIONS_FILE = "./data/actions.csv"

# Load the trained TensorFlow model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load and preprocess the merged data
try:
    loader = DataLoader(CUSTOMERS_FILE, NONCUSTOMERS_FILE, ACTIONS_FILE)
    merged_data = loader.load_and_preprocess()
    print("Merged data loaded successfully.")
except Exception as e:
    print(f"Error loading merged data: {e}")
    merged_data = None

@app.get("/")
def health_check():
    """
    Health check endpoint to verify the app is running.
    """
    return {"status": "ok"}

@app.post("/predict/")
def predict(data: PredictionRequest):
    """
    Predict endpoint to make predictions using the trained model.
    
    Args:
        data (PredictionRequest): Input data containing a list of IDs for which predictions are needed.
    
    Returns:
        dict: Predictions for the input IDs.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    if merged_data is None:
        raise HTTPException(status_code=500, detail="Merged data is not loaded.")

    # Lookup features for the provided IDs
    try:
        features = merged_data[merged_data["id"].isin(data.ids)].drop(columns=["id", "IS_CUSTOMER"])
        if features.empty:
            raise ValueError("No matching IDs found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during feature lookup: {e}")

    # Convert features to numpy array
    features = features.astype(np.float32).to_numpy()
    print('yes new code')
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Input contains NaN or infinite values.")

    # Check data type
    print(f"Features dtype: {features.dtype}")  # Should be float32

    # Check shape
    print(f"Features shape: {features.shape}")

    try:
        # Make predictions
        tensor_input = tf.convert_to_tensor(features, dtype=tf.float32)
        print("1")
        predictions_prob = model.predict(tensor_input)
        print("2")
        predictions = np.argmax(predictions_prob, axis=1)  # Convert probabilities to class labels
        print("3")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/metrics")
def metrics():
    """
    Metrics endpoint for Prometheus scraping.
    """
    from prometheus_client import generate_latest
    return generate_latest()

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def evaluate_model(model_path: str, val_features_file: str, val_labels_file: str):
    """
    Evaluate the trained TensorFlow model using saved validation data.

    Args:
        model_path (str): Path to the saved TensorFlow model.
        val_features_file (str): Path to the validation features CSV file.
        val_labels_file (str): Path to the validation labels CSV file.

    Returns:
        None
    """
    # Load validation data
    X_val = pd.read_csv(val_features_file)
    y_val = pd.read_csv(val_labels_file).squeeze()  # Ensure y_val is a Series

    # Load the saved model
    model = load_model(model_path)

    # Generate predictions
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="binary")
    recall = recall_score(y_val, y_pred, average="binary")
    f1 = f1_score(y_val, y_pred, average="binary")

    # Print metrics
    print(f"Model Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    # File paths
    model_path = "artifacts/tf_model"
    val_features_file = "artifacts/X_val.csv"
    val_labels_file = "artifacts/y_val.csv"

    # Run evaluation
    evaluate_model(model_path, val_features_file, val_labels_file)

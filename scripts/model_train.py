import yaml
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure the project root is in the Python path

# Calculate the script's directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Calculate the parent directory
parent_dir = os.path.dirname(project_root)

# Add the directories to sys.path
sys.path.append(project_root)
sys.path.append(parent_dir)


from preprocess.registry import get_preprocessor
from models.registry import get_model
from data.loader import DataLoader

def main(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    preprocessor_params = config["preprocessor"].get("params", {})

    # Load and preprocess data
    print("Loading data...")
    loader = DataLoader(
        customers_file="data/customers.csv",
        noncustomers_file="data/noncustomers.csv",
        actions_file="data/actions.csv"
    )
    data = loader.load_and_preprocess()
    labels = data["IS_CUSTOMER"]  # Assuming 'IS_CUSTOMER' is the target variable
    data.drop(columns=["IS_CUSTOMER", "id"], inplace=True)  # Remove non-feature columns

    # Get the appropriate preprocessor
    print(f"Initializing preprocessor for model '{model_name}'...")
    preprocessor = get_preprocessor(model_name, **preprocessor_params)
    print("Preprocessing data...")
    processed_data = preprocessor.preprocess(data)

# Get required features after preprocessing
    required_features = preprocessor.get_required_features()
    print(f"Required features: {required_features}")

    # Validate that all required features are present
    missing_features = set(required_features) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocessor.preprocess(data[required_features])

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data, labels, test_size=0.2, random_state=42
    )

    # Convert labels to appropriate format if necessary
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Initialize and train the model
    print(f"Initializing model '{model_name}' with parameters: {model_params}")
    model = get_model(model_name, **model_params)
    print("Training the model...")
    model.train(X_train, y_train, X_val=X_val, y_val=y_val)

    # Evaluate the model on the validation set
    print("Evaluating the model on the validation set...")
    evaluation_metrics = model.evaluate(X_val, y_val)
    print(f"Evaluation metrics: {evaluation_metrics}")

    # Save the trained model
    model_artifact_path = os.path.join("artifacts", model_name + "_model")
    print(f"Saving the model to '{model_artifact_path}'...")
    model.save(model_artifact_path)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args.config)

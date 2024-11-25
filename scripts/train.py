import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




from data.loader import DataLoader
from models.tensorflow_model import TensorFlowModel
from sklearn.model_selection import train_test_split

# File paths
customers_file = "./data/customers.csv"
noncustomers_file = "./data/noncustomers.csv"
actions_file = "./data/actions.csv"

# Load and preprocess data
loader = DataLoader(customers_file, noncustomers_file, actions_file)
merged_data = loader.load_and_preprocess()

# Split data into features and labels
X = merged_data.drop(columns=["IS_CUSTOMER", "id"])  # Features
y = merged_data["IS_CUSTOMER"]                 # Labels

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val.to_csv("artifacts/X_val.csv", index=False)
y_val.to_csv("artifacts/y_val.csv", index=False)

# Initialize TensorFlow model
model = TensorFlowModel(input_shape=(X_train.shape[1],), num_classes=2)

# Train the model
model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

# Save the trained model
model.save("artifacts/tf_model")

import pickle
from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        """
        Initialize the Logistic Regression model with optional parameters.
        """
        self.model = LogisticRegression(**kwargs)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the Logistic Regression model.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model on validation/test data.
        """
        accuracy = self.model.score(X, y)
        return {"accuracy": accuracy}

    def predict(self, X, **kwargs):
        """
        Make predictions on new data.
        """
        return self.model.predict(X)

    def save(self, filepath):
        """
        Save the model to a file using pickle.
        """
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        """
        Load the model from a pickle file.
        """
        with open(f"{filepath}.pkl", "rb") as f:
            self.model = pickle.load(f)

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all models in the framework. 
    This ensures that every model implements a consistent interface.
    """
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            **kwargs: Additional training parameters.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model on validation/test data.
        
        Args:
            X: Features to evaluate on.
            y: Labels to evaluate on.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            Performance metrics.
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on.
            **kwargs: Additional prediction parameters.
            
        Returns:
            Predictions.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from.
        """
        pass

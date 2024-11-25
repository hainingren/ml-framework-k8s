from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors. All custom preprocessors should inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def preprocess(self, data):
        """
        Preprocess the data according to the model's requirements.

        Args:
            data (pandas.DataFrame): The raw input data.

        Returns:
            The preprocessed data ready for model consumption.
        """
        pass

    @abstractmethod
    def get_required_features(self):
        """
        Return a list of feature names required by the model.

        Returns:
            List[str]: A list of required feature names.
        """
        pass

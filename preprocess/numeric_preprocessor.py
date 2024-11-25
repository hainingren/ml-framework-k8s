# preprocess/numeric_preprocessor.py

from preprocess.base_preprocessor import BasePreprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class NumericPreprocessor(BasePreprocessor):
    """
    Preprocessor for models that require numerical features, such as Logistic Regression.
    """

    def __init__(self):
        """
        Initialize the preprocessor.
        """
        self.scaler = StandardScaler()
        self.numerical_features = None

    def preprocess(self, data):
        """
        Scale numerical features using StandardScaler.

        Args:
            data (pandas.DataFrame): The raw input data.

        Returns:
            numpy.ndarray: The preprocessed numerical data.
        """
        # Infer numerical features if not already done
        if self.numerical_features is None:
            self.numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        print(self.numerical_features)
        # Not allowing Nan
        data = data.fillna(-1)
        # Select numerical features

        numerical_data = data[self.numerical_features]
        
        # Fit and transform the data
        scaled_data = self.scaler.fit_transform(numerical_data)

        return scaled_data

    def get_required_features(self):
        """
        Return the list of numerical feature names.

        Returns:
            List[str]: List of numerical feature names.
        """
        # Since features are inferred during preprocessing, return them here
        return self.numerical_features if self.numerical_features else []

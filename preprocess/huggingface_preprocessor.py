# preprocess/huggingface_preprocessor.py

from preprocess.base_preprocessor import BasePreprocessor
from transformers import AutoTokenizer
import pandas as pd

class HuggingFacePreprocessor(BasePreprocessor):
    """
    Preprocessor for Hugging Face transformer models.
    """

    def __init__(self, model_name='bert-base-uncased', max_length=128):
        """
        Initialize the tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to use for tokenization.
            max_length (int): The maximum sequence length for tokenization.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess(self, data):
        """
        Tokenize the text data.

        Args:
            data (pandas.DataFrame): Data containing the 'text' column.

        Returns:
            A dictionary containing tokenized inputs.
        """
        # Ensure 'text' column exists
        if 'text' not in data.columns:
            raise ValueError("Input data must contain a 'text' column.")
        
        texts = data['text'].astype(str).tolist()
        encoded_inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'  # Return PyTorch tensors
        )
        return encoded_inputs

    def get_required_features(self):
        """
        Returns the list of features required by the preprocessor.

        Returns:
            List[str]: ['text']
        """
        return ['text']

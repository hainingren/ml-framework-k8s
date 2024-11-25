#from preprocess.huggingface_preprocessor import HuggingFacePreprocessor
#from preprocess.tree_preprocessor import TreePreprocessor
from preprocess.numeric_preprocessor import NumericPreprocessor

# Registry mapping model names to their respective preprocessors
PREPROCESSOR_REGISTRY = {
    #"huggingface": HuggingFacePreprocessor,
    #"tree": TreePreprocessor,
    "logistic_regression": NumericPreprocessor,
    "tensorflow": NumericPreprocessor,  # Assuming TensorFlow models use numerical data
}

def get_preprocessor(model_name, **kwargs):
    """
    Retrieve the appropriate preprocessor for the given model name.

    Args:
        model_name (str): The name of the model for which to get the preprocessor.
        **kwargs: Additional keyword arguments to pass to the preprocessor's constructor.

    Returns:
        An instance of the preprocessor class corresponding to the model name.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    if model_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"Preprocessor for model '{model_name}' not found in registry.")
    preprocessor_class = PREPROCESSOR_REGISTRY[model_name]
    return preprocessor_class(**kwargs)

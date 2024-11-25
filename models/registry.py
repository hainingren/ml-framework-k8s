from models.tensorflow_model import TensorFlowModel
#from models.huggingface_model import HuggingFaceModel
from models.sklearn_logistic_model import LogisticRegressionModel

MODEL_REGISTRY = {
    "tensorflow": TensorFlowModel,
   # "huggingface": HuggingFaceModel,
    "logistic_regression": LogisticRegressionModel,
}

def get_model(model_name, **kwargs):
    """
    Fetch the appropriate model from the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
    return MODEL_REGISTRY[model_name](**kwargs)
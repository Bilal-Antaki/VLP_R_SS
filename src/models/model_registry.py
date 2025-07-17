# src/models/model_registry.py
from .linear import build_linear_model
from .svr import build_svr_optimized
from .lstm import build_lstm_model
from .gru import build_gru_model
from .cnn import build_cnn_model

MODEL_REGISTRY = {
    # Linear models
    "linear": build_linear_model,
    
    # SVR models
    "svr": build_svr_optimized,
    
    # Neural network models
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "cnn": build_cnn_model,
}

def get_model(name: str, **kwargs):
    """
    Get a model from the registry
    
    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Model-specific parameters
        
    Returns:
        Configured model instance
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available_models = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{name}' not found in registry. Available models: {available_models}")
    
    return MODEL_REGISTRY[name](**kwargs)

def list_available_models():
    """List all available models grouped by category"""
    categories = {
        'Linear': ['linear'],
        'SVM': ['svr'],
        'Neural Networks': ['lstm', 'gru', 'cnn']
    }
    
    print("Available Models by Category:")
    print("=" * 50)
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            if model in MODEL_REGISTRY:
                print(f"  - {model}")
    
    return list(MODEL_REGISTRY.keys())
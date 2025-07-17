from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..config import MODEL_CONFIG, LINEAR_MODEL_TYPE
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor

def build_linear_model(model_type=None, **kwargs):
    if model_type is None:
        model_type = LINEAR_MODEL_TYPE
    model_type = model_type.lower()
    # Map model_type to sklearn class and config
    model_map = {
        'linear': (LinearRegression, MODEL_CONFIG.get('linear', {})),
        'ridge': (Ridge, MODEL_CONFIG.get('ridge', {})),
        'lasso': (Lasso, MODEL_CONFIG.get('lasso', {})),
        'elasticnet': (ElasticNet, MODEL_CONFIG.get('elasticnet', {})),
        'bayesianridge': (BayesianRidge, MODEL_CONFIG.get('bayesianridge', {})),
        'huber': (HuberRegressor, MODEL_CONFIG.get('huber', {})),
        'sgd': (SGDRegressor, MODEL_CONFIG.get('sgd', {})),
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown linear model type: {model_type}")
    ModelClass, config = model_map[model_type]
    # Allow kwargs to override config
    params = {**config, **kwargs}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('linear', ModelClass(**params))
    ])

def get_linear_model(model_type=None, **kwargs):
    return build_linear_model(model_type=model_type, **kwargs)
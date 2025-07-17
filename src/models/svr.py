from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..config import MODEL_CONFIG
from sklearn.svm import SVR

def build_svr_model(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', **kwargs):
    """
    Build SVR model with different kernels and parameters
    
    Args:
        kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        **kwargs: Additional SVR parameters
    """
    # Use config defaults unless overridden
    svr_config = MODEL_CONFIG.get('svr', {})
    svr_params = {
        'kernel': kwargs.pop('kernel', kernel if kernel != 'rbf' else svr_config.get('kernel', 'rbf')),
        'C': kwargs.pop('C', C if C != 1.0 else svr_config.get('C', 1.0)),
        'epsilon': kwargs.pop('epsilon', epsilon if epsilon != 0.1 else svr_config.get('epsilon', 0.1)),
        'gamma': kwargs.pop('gamma', gamma if gamma != 'scale' else svr_config.get('gamma', 'scale')),
        **kwargs
    }
    
    # SVR benefits greatly from feature scaling, so include it in pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ])

def get_svr_model(**kwargs):
    return build_svr_model(**kwargs)
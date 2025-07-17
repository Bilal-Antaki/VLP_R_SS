# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, mean_absolute_percentage_error
)

def calculate_all_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive metrics for regression evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    # Additional metrics
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    
    # Custom metrics
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Percentile errors
    p50_error = np.percentile(abs_errors, 50)  # Median absolute error
    p90_error = np.percentile(abs_errors, 90)
    p95_error = np.percentile(abs_errors, 95)
    
    # Maximum error
    max_error = np.max(abs_errors)
    
    # Relative errors
    relative_errors = abs_errors / (y_true + 1e-10)  # Add small value to avoid division by zero
    mean_relative_error = np.mean(relative_errors) * 100
    
    # Create metrics dictionary
    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': ev,
        'mape': mape,
        'median_abs_error': p50_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        'max_error': max_error,
        'mean_relative_error': mean_relative_error,
        'residual_std': np.std(errors)
    }
    
    return metrics
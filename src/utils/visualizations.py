import matplotlib.pyplot as plt
import numpy as np
import os

def plot_actual_vs_estimated(y_true, y_pred, model_name="Model", save_dir="results/plots"):
    """
    Create and save scatter plots of actual vs estimated r values
    
    Args:
        y_true: Array of actual r values
        y_pred: Array of predicted r values
        model_name: Name of the model for the plot title
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label=f"{model_name} Predictions")
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Add labels and title
    plt.xlabel("Actual r")
    plt.ylabel("Estimated r")
    plt.title(f"{model_name}: Actual vs Estimated r Values\nRMSE: {rmse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_actual_vs_estimated.png"))
    plt.close()
    
    return rmse

def plot_rmse_comparison(model_results, save_dir="results/plots"):
    """
    Create and save a bar plot comparing RMSE values across all models
    
    Args:
        model_results: List of dictionaries containing model results
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort results by RMSE for better visualization
    sorted_results = sorted(model_results, key=lambda x: x['metrics']['rmse'])
    
    # Extract model names and RMSE values
    model_names = [r['name'] for r in sorted_results]
    rmse_values = [r['metrics']['rmse'] for r in sorted_results]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.barh(model_names, rmse_values)
    
    plt.xlabel('RMSE')
    plt.title('Model Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add RMSE values as text on the bars
    for i, v in enumerate(rmse_values):
        plt.text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, "rmse_comparison.png"))
    plt.close()
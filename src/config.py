"""
Configuration settings for the Position Estimation project
"""

# Model Configuration
MODEL_CONFIG = {
    'hidden_size': 256,        # Increased hidden size
    'num_layers': 3,           # Increased number of layers
    'dropout': 0.4,            # Increased dropout for better regularization
    'sequence_length': 16,     # Increased sequence length for better temporal patterns
    'batch_size': 64,          # Increased batch size for better training stability
    'learning_rate': 0.001,    # Reduced learning rate for better convergence
    'epochs': 200,             # Increased epochs with early stopping
    'patience': 30             # Increased patience for early stopping
}

# GRU Model Configuration
GRU_CONFIG = {
    'hidden_dim': 64,        # Hidden dimension for GRU cells
    'num_layers': 2,          # Number of GRU layers
    'dropout': 0.3           # Dropout rate between layers
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 300,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'correlation_threshold': 0.1,
    'simulation_length': 10,
    'base_features': ['PL', 'RMS']
}

# Data Processing Configuration
DATA_CONFIG = {
    'input_file': 'data/processed/FCPR-D1_CIR.csv',
    'target_column': 'r',
    'processed_dir': 'data/processed',
    'datasets': ['FCPR-D1']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'feature_selection': {
        'correlation_threshold': 0.3,
        'excluded_features': ['r', 'X', 'Y']
    },
    'visualization': {
        'figure_sizes': {
            'data_exploration': (12, 5),
            'model_comparison': (17, 6)
        },
        'height_ratios': [1, 1],
        'scatter_alpha': 0.6,
        'scatter_size': 20,
        'grid_alpha': 0.3
    },
    'output': {
        'results_dir': 'results',
        'report_file': 'analysis_report.txt'
    }
}

# Model Training Options
TRAINING_OPTIONS = {
    'include_slow_models': True,  # Whether to include computationally intensive models
    'save_predictions': True,      # Whether to save model predictions
    'plot_training_history': True  # Whether to plot training history for applicable models
}

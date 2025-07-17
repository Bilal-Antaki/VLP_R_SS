"""
Configuration settings for the Position Estimation project
"""

# Model Configuration - Different parameters for each model type
MODEL_CONFIG = {
    # GRU specific parameters
    'gru': {
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 300,
        'patience': 30
    },
    # LSTM specific parameters
    'lstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 300,
        'patience': 30
    },
    # CNN specific parameters - Updated to prevent constant predictions
    'cnn': {
        'hidden_size': 64,  # Increased from 32
        'num_layers': 3,    # Increased from 2
        'dropout': 0.4,     # Reduced from 0.5
        'learning_rate': 0.001,  # Reduced from 0.0015
        'batch_size': 32,   # Reduced from 48
        'epochs': 300,      # Increased from 200
        'patience': 25,     # Increased from 20
        'kernel_size': 3,
        'num_filters': 128, # Increased from 64
        'weight_decay': 1e-5,  # Added for optimizer
        'scheduler_patience': 10  # Added for LR scheduler
    },
    # Common parameters
    'sequence_length': 10
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 300,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42
}

# Data Configuration
DATA_CONFIG = {
    'input_file': 'data/processed/FCPR-D1_CIR.csv',
    'target_column': 'r',
    'processed_dir': 'data/processed',
    'datasets': ['FCPR-D1']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'visualization': {
        'figure_sizes': {
            'data_exploration': (12, 5),
            'model_comparison': (17, 6)
        },
        'height_ratios': [1],
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
    'save_predictions': True,
    'plot_training_history': True
}

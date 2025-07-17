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
    # SVR specific parameters
    'svr': {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    },
    # Ridge Regression specific parameters
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True,
        'solver': 'auto',
        'max_iter': None,
        'tol': 1e-3
    },
    # Lasso Regression specific parameters
    'lasso': {
        'alpha': 1.0,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-4,
        'selection': 'cyclic'
    },
    # ElasticNet Regression specific parameters
    'elasticnet': {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-4,
        'selection': 'cyclic'
    },
    # Bayesian Ridge Regression specific parameters
    'bayesianridge': {
        'n_iter': 300,
        'tol': 1e-3,
        'alpha_1': 1e-6,
        'alpha_2': 1e-6,
        'lambda_1': 1e-6,
        'lambda_2': 1e-6,
        'fit_intercept': True,
        'compute_score': False
    },
    # Huber Regressor specific parameters
    'huber': {
        'epsilon': 1.35,
        'alpha': 0.0001,
        'fit_intercept': True,
        'max_iter': 100,
        'tol': 1e-5
    },
    # SGD Regressor specific parameters
    'sgd': {
        'loss': 'squared_loss',
        'penalty': 'l2',
        'alpha': 0.0001,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-3,
        'learning_rate': 'invscaling',
        'eta0': 0.01,
        'early_stopping': False
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

# Top-level option to select which linear regression model to use
LINEAR_MODEL_TYPE = 'linear'  # Options: 'linear', 'ridge', 'lasso', 'elasticnet', 'bayesianridge', 'huber', 'sgd'

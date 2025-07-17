# train_linear.py
# Script to train and evaluate the Linear Regression model

from src.models.linear import get_linear_model
from src.data.data_loader import load_data
from src.evaluation.metrics import calculate_all_metrics
import joblib
import os
from src.config import LINEAR_MODEL_TYPE

def main(model_type=LINEAR_MODEL_TYPE):
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    print(f"Using linear regression model type: {model_type}")
    model = get_linear_model(model_type=model_type)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    results = calculate_all_metrics(y_test, y_pred, model_name=model_type)
    print('Evaluation Results:', results)

    # Save model
    os.makedirs('results/models', exist_ok=True)
    joblib.dump({
        'model': model,
        'metrics': results,
        'predictions': {
            'actual': y_test,
            'predicted': y_pred
        }
    }, 'results/models/linear_model.pkl')
    print('Model and results saved to results/models/linear_model.pkl')

if __name__ == '__main__':
    main() 
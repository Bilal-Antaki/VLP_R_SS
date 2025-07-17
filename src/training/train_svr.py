# train_svr.py
# Script to train and evaluate the SVR model

from src.models.svr import get_svr_model
from src.data.data_loader import load_data
from src.evaluation.metrics import calculate_all_metrics
import joblib
import os

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Get model
    model = get_svr_model()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    results = calculate_all_metrics(y_test, y_pred, model_name='svr')
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
    }, 'results/models/svr_model.pkl')
    print('Model and results saved to results/models/svr_model.pkl')

if __name__ == '__main__':
    main() 
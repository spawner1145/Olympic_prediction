from preprocessing import preprocess_data
from random_forest_model import train_and_evaluate_rf
from deep_learning_model import train_and_evaluate_dl
import numpy as np
import torch

def main():
    # Filepath to your data
    filepath = 'summerOly_medal_counts.xlsx'

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(filepath)

    # Train and evaluate the Random Forest model
    rf_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test)

    # Train and evaluate the Deep Learning model
    dl_model = train_and_evaluate_dl(X_train, X_test, y_train, y_test)

    # Combine predictions (simple average)
    rf_preds = rf_model.predict(X_test)
    dl_preds, _ = dl_model(torch.tensor(X_test, dtype=torch.float32))
    dl_preds = dl_preds.detach().numpy().flatten()
    ensemble_preds = (rf_preds + dl_preds) / 2

    # Evaluate ensemble predictions
    mae = np.mean(np.abs(ensemble_preds - y_test))
    rmse = np.sqrt(np.mean((ensemble_preds - y_test)**2))
    r2 = 1 - (np.sum((y_test - ensemble_preds)**2) / np.sum((y_test - np.mean(y_test))**2))

    print("Ensemble Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 Score: {r2:.2f}")

if __name__ == "__main__":
    main()
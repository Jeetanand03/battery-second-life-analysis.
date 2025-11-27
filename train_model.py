import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

def train_model(X, y, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = XGBRegressor(random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        scores = evaluate(y.iloc[val_idx], pred)
        all_scores.append(scores)
        print(f"{label} Fold {fold+1}: RMSE={scores[0]:.4f}, MAE={scores[1]:.4f}, MAPE={scores[2]:.2f}%")

    os.makedirs("models", exist_ok=True)
    model.fit(X, y)
    joblib.dump(model, f"models/{label}_model.pkl")
    print(f"Saved model: models/{label}_model.pkl")

def main():
    df = pd.read_csv("data/processed/features.csv")
    X = df[['v_mean', 'i_peak', 'energy']]
    train_model(X, df['soh'], 'soh')
    train_model(X, df['sop'], 'sop')

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_wine_quality_model():
    print("Fetching wine quality dataset...")
    # Using red wine dataset from UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        df = pd.read_csv(url, sep=';')
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/winequality-red.csv", index=False)
        print("Dataset saved to data/winequality-red.csv")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Features and Target
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross Validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    metrics = {
        "Mean Squared Error (MSE)": round(mse, 4),
        "Root Mean Squared Error (RMSE)": round(rmse, 4),
        "Mean Absolute Error (MAE)": round(mae, 4),
        "R-squared (R2)": round(r2, 4),
        "Cross-Validation R2 (Mean)": round(cv_scores.mean(), 4)
    }

    print("\nModel Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Feature Importance
    feature_importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    metrics["feature_importance"] = feature_importance

    # Save outputs
    joblib.dump(model, "wine_quality_model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining complete. Model and metrics saved.")

if __name__ == "__main__":
    train_wine_quality_model()

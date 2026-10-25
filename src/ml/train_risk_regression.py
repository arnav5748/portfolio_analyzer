
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "forecasting"

OUTPUT_DIR = BASE_DIR / "models" / "regression"
RESULTS_DIR = BASE_DIR / "results" / "regression"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

feature_columns = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "annualized_portfolio_volatility",
    "portfolio_drawdown",
    "mkt_rf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "mom",
    "rf",
    "CPI",
    "FEDERAL_FUNDS_RATE",
    "TREASURY_10Y",
    "UNEMPLOYMENT_RATE",
    "VIX"
]

target_column = "future_volatility_10d"

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_val = val_df[feature_columns]
y_val = val_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

print("Training observations:", len(X_train))
print("Validation observations:", len(X_val))
print("Test observations:", len(X_test))


# --------------------------------------------------
# 3. DEFINE REGRESSION MODELS
# --------------------------------------------------

models = {
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# 4. TRAIN AND VALIDATE
# --------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    # Volatility cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print("Validation MAE:", round(mae, 6))
    print("Validation RMSE:", round(rmse, 6))
    print("Validation R2:", round(r2, 6))


# --------------------------------------------------
# 5. SELECT MODEL USING VALIDATION MAE
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values("MAE")

print("\nVALIDATION MODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS_DIR / "regression_model_comparison.csv",
    index=False
)

best_model_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 6. FINAL TEST EVALUATION
# --------------------------------------------------

test_predictions = best_model.predict(X_test)

test_predictions = np.maximum(test_predictions, 0)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("\nFINAL TEST RESULTS")
print("MAE:", round(test_mae, 6))
print("RMSE:", round(test_rmse, 6))
print("R2:", round(test_r2, 6))


# --------------------------------------------------
# 7. CONVERT REGRESSION PREDICTIONS TO RISK ALERTS
# --------------------------------------------------

# Use the rolling threshold already calculated
# during forecasting dataset preparation.

test_threshold = test_df["risk_threshold"].values

actual_risk = (
    y_test.values >= test_threshold
).astype(int)

predicted_risk = (
    test_predictions >= test_threshold
).astype(int)

precision = precision_score(
    actual_risk,
    predicted_risk,
    zero_division=0
)

recall = recall_score(
    actual_risk,
    predicted_risk,
    zero_division=0
)

f1 = f1_score(
    actual_risk,
    predicted_risk,
    zero_division=0
)

pr_auc = average_precision_score(
    actual_risk,
    test_predictions
)

print("\nRISK ALERT PERFORMANCE")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual_risk,
        predicted_risk,
        labels=[0, 1]
    )
)


# --------------------------------------------------
# 8. SAVE PREDICTIONS
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "trading_date": test_df["trading_date"],
    "actual_future_volatility": y_test.values,
    "predicted_future_volatility": test_predictions,
    "risk_threshold": test_threshold,
    "actual_high_risk": actual_risk,
    "predicted_high_risk": predicted_risk
})

predictions_df.to_csv(
    RESULTS_DIR / "test_regression_predictions.csv",
    index=False
)


# --------------------------------------------------
# 9. SAVE METRICS AND MODEL
# --------------------------------------------------

metrics_df = pd.DataFrame([{
    "Selected Model": best_model_name,
    "MAE": test_mae,
    "RMSE": test_rmse,
    "R2": test_r2,
    "Risk Precision": precision,
    "Risk Recall": recall,
    "Risk F1": f1,
    "Risk PR-AUC": pr_auc
}])

metrics_df.to_csv(
    RESULTS_DIR / "final_regression_metrics.csv",
    index=False
)

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_regression_model.pkl"
)

print("\nRegression pipeline completed!")
print("Model saved to:", OUTPUT_DIR)
print("Results saved to:", RESULTS_DIR)
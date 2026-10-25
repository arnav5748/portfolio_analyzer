
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results" / "regression"

df = pd.read_csv(
    RESULTS_DIR / "test_regression_predictions.csv"
)

actual = df["actual_future_volatility"]
predicted = df["predicted_future_volatility"]

# Prediction errors
df["error"] = predicted - actual
df["absolute_error"] = df["error"].abs()

print("\nPREDICTION ANALYSIS")
print("-------------------")

print("Actual volatility mean:", actual.mean())
print("Predicted volatility mean:", predicted.mean())

print("Actual volatility std:", actual.std())
print("Predicted volatility std:", predicted.std())

print("MAE:", mean_absolute_error(actual, predicted))
print("R2:", r2_score(actual, predicted))

print("\nLargest prediction errors:")
print(
    df.sort_values(
        "absolute_error",
        ascending=False
    ).head(10).to_string(index=False)
)

# Actual vs predicted plot
plt.figure(figsize=(10, 6))

plt.plot(
    actual.values,
    label="Actual volatility"
)

plt.plot(
    predicted.values,
    label="Predicted volatility"
)

plt.title("Actual vs Predicted Future Volatility")
plt.xlabel("Test observation")
plt.ylabel("Future volatility")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# Scatter plot
plt.figure(figsize=(7, 7))

plt.scatter(actual, predicted)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual volatility")
plt.ylabel("Predicted volatility")
plt.title("Actual vs Predicted Volatility")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "regression_scatter.png",
    dpi=300
)

plt.show()

print("\nAnalysis completed!")


# --------------------------------------------------
# BASELINE COMPARISON
# --------------------------------------------------

train_df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "forecasting" / "train.csv"
)

baseline_prediction = np.full(
    len(actual),
    train_df["future_volatility_10d"].mean()
)

baseline_mae = mean_absolute_error(
    actual,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    np.mean((actual - baseline_prediction) ** 2)
)

baseline_r2 = r2_score(
    actual,
    baseline_prediction
)

print("\nBASELINE VS RANDOM FOREST")
print("-------------------------")

print("Baseline MAE:", round(baseline_mae, 6))
print("Random Forest MAE:", round(
    mean_absolute_error(actual, predicted), 6
))

print("Baseline RMSE:", round(baseline_rmse, 6))
print("Random Forest RMSE:", round(
    np.sqrt(np.mean((actual - predicted) ** 2)), 6
))

print("Baseline R2:", round(baseline_r2, 6))
print("Random Forest R2:", round(
    r2_score(actual, predicted), 6
))

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "portfolio_ml_dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "forecasting"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["trading_date"] = pd.to_datetime(df["trading_date"])
df = df.sort_values("trading_date").reset_index(drop=True)

print("Original dataset:", df.shape)

# --------------------------------------------------
# 3. FEATURES AND TARGET SOURCE
# --------------------------------------------------

feature_columns = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "annualized_portfolio_volatility",
    "portfolio_drawdown",
    "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf",
    "CPI",
    "FEDERAL_FUNDS_RATE",
    "TREASURY_10Y",
    "UNEMPLOYMENT_RATE",
    "VIX"
]

target_source = "future_volatility_10d"

# Remove rows with missing required data
df = df.dropna(
    subset=feature_columns + [target_source]
).copy()

df = df.reset_index(drop=True)

# --------------------------------------------------
# 4. ROLLING HISTORICAL THRESHOLD
# --------------------------------------------------

# Historical future-volatility observations are shifted
# by 10 trading days to avoid using outcomes that would
# not yet have been observable at the forecast date.

historical_volatility = df[target_source].shift(10)

df["risk_threshold"] = (
    historical_volatility
    .rolling(window=252, min_periods=126)
    .quantile(0.90)
)

# Create binary target
df["high_future_risk"] = np.where(
    df["risk_threshold"].notna(),
    (
        df[target_source] >= df["risk_threshold"]
    ).astype(int),
    np.nan
)

# Remove initial rows without enough historical data
df = df.dropna(
    subset=["risk_threshold", "high_future_risk"]
).copy()

df["high_future_risk"] = df["high_future_risk"].astype(int)

df = df.reset_index(drop=True)

print("\nDataset after rolling threshold:", df.shape)

# --------------------------------------------------
# 5. CHRONOLOGICAL SPLIT
# --------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# --------------------------------------------------
# 6. CLASS DISTRIBUTION
# --------------------------------------------------

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")
    print("Rows:", len(part))
    print(part["high_future_risk"].value_counts().sort_index())
    print(
        part["high_future_risk"]
        .value_counts(normalize=True)
        .sort_index()
    )

# --------------------------------------------------
# 7. PREPARE X AND Y
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["high_future_risk"]

X_val = val_df[feature_columns]
y_val = val_df["high_future_risk"]

X_test = test_df[feature_columns]
y_test = test_df["high_future_risk"]

# --------------------------------------------------
# 8. SAVE DATASETS
# --------------------------------------------------

combined_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

combined_df.to_csv(
    OUTPUT_DIR / "forecasting_dataset.csv",
    index=False
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)

X_val.to_csv(OUTPUT_DIR / "X_validation.csv", index=False)
y_val.to_csv(OUTPUT_DIR / "y_validation.csv", index=False)

X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Save threshold history for auditability
df[[
    "trading_date",
    target_source,
    "risk_threshold",
    "high_future_risk"
]].to_csv(
    OUTPUT_DIR / "risk_threshold_history.csv",
    index=False
)

print("\nForecasting dataset preparation completed!")
print("Features:", len(feature_columns))
print("Files saved to:", OUTPUT_DIR)
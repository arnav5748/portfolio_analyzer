import pandas as pd
import os


INPUT_FILE = "data/processed/portfolio_ml_dataset.csv"
OUTPUT_DIR = "data/processed/regime"


# Features used for market regime detection
REGIME_FEATURES = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "portfolio_drawdown",
    "VIX",
    "mkt_rf",
    "TREASURY_10Y"
]


# ---------------------------------------------------------
# 1. Load portfolio dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["trading_date"] = pd.to_datetime(df["trading_date"])

print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Select regime features
# ---------------------------------------------------------

regime_data = df[
    ["trading_date"] + REGIME_FEATURES
].copy()


# ---------------------------------------------------------
# 3. Remove rows with missing regime features
# ---------------------------------------------------------

regime_data = regime_data.dropna(
    subset=REGIME_FEATURES
).copy()


# ---------------------------------------------------------
# 4. Sort chronologically
# ---------------------------------------------------------

regime_data = (
    regime_data
    .sort_values("trading_date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 5. Create chronological train/validation/test split
# ---------------------------------------------------------

n = len(regime_data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = regime_data.iloc[:train_end].copy()

validation = regime_data.iloc[
    train_end:validation_end
].copy()

test = regime_data.iloc[
    validation_end:
].copy()


# ---------------------------------------------------------
# 6. Create output directory
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 7. Save datasets
# ---------------------------------------------------------

regime_data.to_csv(
    f"{OUTPUT_DIR}/regime_dataset.csv",
    index=False
)

train.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

validation.to_csv(
    f"{OUTPUT_DIR}/validation.csv",
    index=False
)

test.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)


# ---------------------------------------------------------
# 8. Print summary
# ---------------------------------------------------------

print("\n--------------------------------")
print("REGIME DATASET READY")
print("--------------------------------")

print("\nFinal dataset shape:")
print(regime_data.shape)

print("\nFeatures:")
print(REGIME_FEATURES)

print("\nMissing values:")
print(regime_data.isnull().sum())

print("\nDate range:")
print(
    regime_data["trading_date"].min(),
    "to",
    regime_data["trading_date"].max()
)

print("\nTrain:")
print(train.shape)

print("Validation:")
print(validation.shape)

print("Test:")
print(test.shape)

print("\nTrain dates:")
print(
    train["trading_date"].min(),
    "to",
    train["trading_date"].max()
)

print("\nValidation dates:")
print(
    validation["trading_date"].min(),
    "to",
    validation["trading_date"].max()
)

print("\nTest dates:")
print(
    test["trading_date"].min(),
    "to",
    test["trading_date"].max()
)

print("\nSaved to:")
print(OUTPUT_DIR)
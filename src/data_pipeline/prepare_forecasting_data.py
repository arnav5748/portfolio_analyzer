import pandas as pd
import os


INPUT_FILE = "data/processed/portfolio_ml_dataset.csv"
OUTPUT_DIR = "data/processed/forecasting"


FEATURE_COLUMNS = [
    # Portfolio features
    "portfolio_return",
    "portfolio_volatility_20d",
    "annualized_portfolio_volatility",
    "portfolio_drawdown",

    # Fama-French factors
    "mkt_rf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "mom",
    "rf",

    # Macro indicators
    "CPI",
    "FEDERAL_FUNDS_RATE",
    "TREASURY_10Y",
    "UNEMPLOYMENT_RATE",
    "VIX"
]

TARGET_COLUMN = "high_future_risk"


# Load dataset
df = pd.read_csv(INPUT_FILE)

df["trading_date"] = pd.to_datetime(df["trading_date"])

print("Original dataset shape:", df.shape)


# Keep date + ML features + target
ml_data = df[
    ["trading_date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
].copy()


# Remove rows where target is unavailable
# or any required feature is missing
ml_data = ml_data.dropna(
    subset=FEATURE_COLUMNS + [TARGET_COLUMN]
).copy()


# Target should be integer 0/1
ml_data[TARGET_COLUMN] = ml_data[TARGET_COLUMN].astype(int)


# Sort chronologically
ml_data = ml_data.sort_values("trading_date").reset_index(drop=True)


print("\nClean forecasting dataset shape:")
print(ml_data.shape)

print("\nDate range:")
print(
    ml_data["trading_date"].min(),
    "to",
    ml_data["trading_date"].max()
)

print("\nMissing values:")
print(ml_data.isnull().sum())

print("\nTarget distribution:")
print(ml_data[TARGET_COLUMN].value_counts())


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Full forecasting dataset
# ---------------------------------------------------------

ml_data.to_csv(
    f"{OUTPUT_DIR}/forecasting_dataset.csv",
    index=False
)


# ---------------------------------------------------------
# Chronological train / validation / test split
# 70% / 15% / 15%
# ---------------------------------------------------------

n = len(ml_data)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = ml_data.iloc[:train_end].copy()
validation = ml_data.iloc[train_end:val_end].copy()
test = ml_data.iloc[val_end:].copy()


# Save full splits
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
# Separate X and y for Teammate 2
# ---------------------------------------------------------

X_train = train[FEATURE_COLUMNS]
y_train = train[TARGET_COLUMN]

X_validation = validation[FEATURE_COLUMNS]
y_validation = validation[TARGET_COLUMN]

X_test = test[FEATURE_COLUMNS]
y_test = test[TARGET_COLUMN]


X_train.to_csv(
    f"{OUTPUT_DIR}/X_train.csv",
    index=False
)

y_train.to_csv(
    f"{OUTPUT_DIR}/y_train.csv",
    index=False
)

X_validation.to_csv(
    f"{OUTPUT_DIR}/X_validation.csv",
    index=False
)

y_validation.to_csv(
    f"{OUTPUT_DIR}/y_validation.csv",
    index=False
)

X_test.to_csv(
    f"{OUTPUT_DIR}/X_test.csv",
    index=False
)

y_test.to_csv(
    f"{OUTPUT_DIR}/y_test.csv",
    index=False
)


print("\n--------------------------------")
print("FORECASTING DATA READY")
print("--------------------------------")

print("\nTrain:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

print("\nTrain dates:")
print(train["trading_date"].min(), "to", train["trading_date"].max())

print("\nValidation dates:")
print(
    validation["trading_date"].min(),
    "to",
    validation["trading_date"].max()
)

print("\nTest dates:")
print(test["trading_date"].min(), "to", test["trading_date"].max())

print("\nFiles saved in:")
print(OUTPUT_DIR)
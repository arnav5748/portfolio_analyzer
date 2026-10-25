import pandas as pd
import os
import joblib


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

REGIME_DIR = "data/processed/regime"
OUTPUT_DIR = "data/processed/regime/hmm"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Features
# ---------------------------------------------------------

FEATURES = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "portfolio_drawdown",
    "VIX",
    "mkt_rf",
    "TREASURY_10Y"
]


# ---------------------------------------------------------
# 3. Load train / validation / test
# ---------------------------------------------------------

train = pd.read_csv(
    f"{REGIME_DIR}/train.csv"
)

validation = pd.read_csv(
    f"{REGIME_DIR}/validation.csv"
)

test = pd.read_csv(
    f"{REGIME_DIR}/test.csv"
)

for df in [train, validation, test]:
    df["trading_date"] = pd.to_datetime(
        df["trading_date"]
    )


# ---------------------------------------------------------
# 4. Load trained HMM and scaler
# ---------------------------------------------------------

hmm = joblib.load(
    f"{OUTPUT_DIR}/hmm_model.pkl"
)

scaler = joblib.load(
    f"{OUTPUT_DIR}/hmm_scaler.pkl"
)


# ---------------------------------------------------------
# 5. Scale using training scaler
# ---------------------------------------------------------

X_train = scaler.transform(
    train[FEATURES]
)

X_validation = scaler.transform(
    validation[FEATURES]
)

X_test = scaler.transform(
    test[FEATURES]
)


# ---------------------------------------------------------
# 6. Apply HMM
# ---------------------------------------------------------

train["hmm_regime"] = hmm.predict(X_train)

validation["hmm_regime"] = hmm.predict(
    X_validation
)

test["hmm_regime"] = hmm.predict(
    X_test
)


# ---------------------------------------------------------
# 7. Get state probabilities
# ---------------------------------------------------------

train_prob = hmm.predict_proba(X_train)

validation_prob = hmm.predict_proba(
    X_validation
)

test_prob = hmm.predict_proba(X_test)


for i in range(4):

    train[f"hmm_state_{i}_probability"] = (
        train_prob[:, i]
    )

    validation[f"hmm_state_{i}_probability"] = (
        validation_prob[:, i]
    )

    test[f"hmm_state_{i}_probability"] = (
        test_prob[:, i]
    )


# ---------------------------------------------------------
# 8. Combine all periods
# ---------------------------------------------------------

all_regimes = pd.concat(
    [train, validation, test],
    ignore_index=True
)

all_regimes = (
    all_regimes
    .sort_values("trading_date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 9. Save
# ---------------------------------------------------------

train.to_csv(
    f"{OUTPUT_DIR}/train_regimes.csv",
    index=False
)

validation.to_csv(
    f"{OUTPUT_DIR}/validation_regimes.csv",
    index=False
)

test.to_csv(
    f"{OUTPUT_DIR}/test_regimes.csv",
    index=False
)

all_regimes.to_csv(
    f"{OUTPUT_DIR}/all_regimes.csv",
    index=False
)


# ---------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------

summary = (
    all_regimes
    .groupby("hmm_regime")[FEATURES]
    .mean()
)

summary["observations"] = (
    all_regimes["hmm_regime"]
    .value_counts()
    .sort_index()
)

print("\n--------------------------------")
print("HMM REGIMES APPLIED")
print("--------------------------------")

print("\nState summary:")
print(summary.to_string())

print("\nState counts:")
print(
    all_regimes["hmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nDate range:")
print(
    all_regimes["trading_date"].min(),
    "to",
    all_regimes["trading_date"].max()
)

summary.to_csv(
    f"{OUTPUT_DIR}/all_hmm_state_summary.csv"
)

print("\nFiles saved in:")
print(OUTPUT_DIR)
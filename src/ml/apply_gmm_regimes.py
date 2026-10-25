import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

REGIME_DIR = "data/processed/regime"
OUTPUT_DIR = "data/processed/regime/gmm"

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


print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ---------------------------------------------------------
# 4. Prepare features
# ---------------------------------------------------------

X_train = train[FEATURES]
X_validation = validation[FEATURES]
X_test = test[FEATURES]


# ---------------------------------------------------------
# 5. Fit scaler ONLY on training data
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ---------------------------------------------------------
# 6. Fit final 4-regime GMM
# ---------------------------------------------------------
# 4 regimes were selected earlier using BIC.
# ---------------------------------------------------------

gmm = GaussianMixture(
    n_components=4,
    covariance_type="full",
    random_state=42,
    n_init=10
)

gmm.fit(X_train_scaled)


# ---------------------------------------------------------
# 7. Predict regimes
# ---------------------------------------------------------

train["regime"] = gmm.predict(
    X_train_scaled
)

validation["regime"] = gmm.predict(
    X_validation_scaled
)

test["regime"] = gmm.predict(
    X_test_scaled
)


# ---------------------------------------------------------
# 8. Add regime probabilities
# ---------------------------------------------------------

train_prob = gmm.predict_proba(
    X_train_scaled
)

validation_prob = gmm.predict_proba(
    X_validation_scaled
)

test_prob = gmm.predict_proba(
    X_test_scaled
)


for i in range(4):
    train[f"regime_{i}_probability"] = train_prob[:, i]

    validation[f"regime_{i}_probability"] = (
        validation_prob[:, i]
    )

    test[f"regime_{i}_probability"] = (
        test_prob[:, i]
    )


# ---------------------------------------------------------
# 9. Combine all periods
# ---------------------------------------------------------

all_regimes = pd.concat(
    [train, validation, test],
    ignore_index=True
)

all_regimes = all_regimes.sort_values(
    "trading_date"
).reset_index(drop=True)


# ---------------------------------------------------------
# 10. Save datasets
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
# 11. Regime summary
# ---------------------------------------------------------

summary = (
    all_regimes
    .groupby("regime")[FEATURES]
    .mean()
)

summary["observations"] = (
    all_regimes["regime"]
    .value_counts()
    .sort_index()
)

print("\n--------------------------------")
print("GMM REGIMES APPLIED")
print("--------------------------------")

print("\nRegime summary:")
print(summary.to_string())

print("\nRegime counts:")
print(
    all_regimes["regime"]
    .value_counts()
    .sort_index()
)

print("\nDate range:")
print(
    all_regimes["trading_date"].min(),
    "to",
    all_regimes["trading_date"].max()
)


# ---------------------------------------------------------
# 12. Save summary
# ---------------------------------------------------------

summary.to_csv(
    f"{OUTPUT_DIR}/all_regime_summary.csv"
)


# ---------------------------------------------------------
# 13. Save scaler and model
# ---------------------------------------------------------

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}/gmm_scaler.pkl"
)

joblib.dump(
    gmm,
    f"{OUTPUT_DIR}/gmm_model.pkl"
)


print("\nFiles saved in:")
print(OUTPUT_DIR)
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


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
# 3. Load training data
# ---------------------------------------------------------

train = pd.read_csv(
    f"{REGIME_DIR}/train.csv"
)

train["trading_date"] = pd.to_datetime(
    train["trading_date"]
)

train = train.sort_values(
    "trading_date"
).reset_index(drop=True)

print("Training data shape:", train.shape)


# ---------------------------------------------------------
# 4. Prepare features
# ---------------------------------------------------------

X = train[FEATURES].copy()


# ---------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 6. Train 4-state HMM
# ---------------------------------------------------------
# GMM previously selected 4 regimes using BIC.
# We use 4 hidden states here for comparison.
# ---------------------------------------------------------

hmm = GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=500,
    random_state=42
)

hmm.fit(X_scaled)


# ---------------------------------------------------------
# 7. Predict hidden states
# ---------------------------------------------------------

train["hmm_regime"] = hmm.predict(
    X_scaled
)


# ---------------------------------------------------------
# 8. State probabilities
# ---------------------------------------------------------

state_probabilities = hmm.predict_proba(
    X_scaled
)

for i in range(4):
    train[f"hmm_state_{i}_probability"] = (
        state_probabilities[:, i]
    )


# ---------------------------------------------------------
# 9. Transition matrix
# ---------------------------------------------------------

transition_matrix = pd.DataFrame(
    hmm.transmat_,
    index=[
        "state_0",
        "state_1",
        "state_2",
        "state_3"
    ],
    columns=[
        "state_0",
        "state_1",
        "state_2",
        "state_3"
    ]
)

print("\nHMM Transition Matrix:")
print(transition_matrix.to_string())


# ---------------------------------------------------------
# 10. State characteristics
# ---------------------------------------------------------

state_summary = (
    train
    .groupby("hmm_regime")[FEATURES]
    .mean()
)

state_summary["observations"] = (
    train["hmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nHMM State Summary:")
print(state_summary.to_string())


# ---------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------

train.to_csv(
    f"{OUTPUT_DIR}/hmm_train_regimes.csv",
    index=False
)

transition_matrix.to_csv(
    f"{OUTPUT_DIR}/transition_matrix.csv"
)

state_summary.to_csv(
    f"{OUTPUT_DIR}/hmm_state_summary.csv"
)


# ---------------------------------------------------------
# 12. Save model and scaler
# ---------------------------------------------------------

joblib.dump(
    hmm,
    f"{OUTPUT_DIR}/hmm_model.pkl"
)

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}/hmm_scaler.pkl"
)


# ---------------------------------------------------------
# 13. Final output
# ---------------------------------------------------------

print("\n--------------------------------")
print("HMM REGIME DETECTION COMPLETE")
print("--------------------------------")

print("\nState counts:")
print(
    train["hmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nLog likelihood:")
print(hmm.score(X_scaled))

print("\nFiles saved in:")
print(OUTPUT_DIR)

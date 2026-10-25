import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

TRAIN_FILE = "data/processed/regime/train.csv"
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
# 3. Load training data
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)

train["trading_date"] = pd.to_datetime(
    train["trading_date"]
)

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
# 6. Try different numbers of regimes
# ---------------------------------------------------------

results = []

for k in [2, 3, 4]:

    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    results.append({
        "n_regimes": k,
        "BIC": model.bic(X_scaled),
        "AIC": model.aic(X_scaled)
    })


results_df = pd.DataFrame(results)

print("\nGMM model comparison:")
print(results_df)

results_df.to_csv(
    f"{OUTPUT_DIR}/gmm_model_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 7. Select model using BIC
# ---------------------------------------------------------

best_k = results_df.loc[
    results_df["BIC"].idxmin(),
    "n_regimes"
]

best_k = int(best_k)

print("\nBest number of regimes according to BIC:", best_k)


# ---------------------------------------------------------
# 8. Train final GMM
# ---------------------------------------------------------

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42,
    n_init=10
)

gmm.fit(X_scaled)


# ---------------------------------------------------------
# 9. Assign regime
# ---------------------------------------------------------

train["regime"] = gmm.predict(X_scaled)


# ---------------------------------------------------------
# 10. Regime probabilities
# ---------------------------------------------------------

probabilities = gmm.predict_proba(X_scaled)

for i in range(best_k):
    train[f"regime_{i}_probability"] = probabilities[:, i]


# ---------------------------------------------------------
# 11. Analyze each regime
# ---------------------------------------------------------

regime_summary = (
    train
    .groupby("regime")[FEATURES]
    .mean()
)

regime_counts = (
    train["regime"]
    .value_counts()
    .sort_index()
    .rename("observations")
)

regime_summary = regime_summary.join(regime_counts)

print("\nRegime summary:")
print(regime_summary.to_string())

regime_summary.to_csv(
    f"{OUTPUT_DIR}/regime_summary.csv"
)


# ---------------------------------------------------------
# 12. Save regime assignments
# ---------------------------------------------------------

train.to_csv(
    f"{OUTPUT_DIR}/train_regimes.csv",
    index=False
)


# ---------------------------------------------------------
# 13. Final output
# ---------------------------------------------------------

print("\n--------------------------------")
print("GMM REGIME DETECTION COMPLETE")
print("--------------------------------")

print("\nNumber of regimes:", best_k)

print("\nRegime counts:")
print(train["regime"].value_counts().sort_index())

print("\nFiles saved in:")
print(OUTPUT_DIR)
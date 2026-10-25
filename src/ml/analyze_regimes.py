import pandas as pd
import matplotlib.pyplot as plt
import os


INPUT_FILE = "data/processed/regime/combined_regimes.csv"
OUTPUT_DIR = "data/processed/regime/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Load combined regime dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["trading_date"] = pd.to_datetime(
    df["trading_date"]
)

df = df.sort_values("trading_date")


print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. GMM vs HMM regime counts
# ---------------------------------------------------------

print("\nGMM regime counts:")
print(
    df["gmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nHMM state counts:")
print(
    df["hmm_regime"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# 3. GMM regime characteristics
# ---------------------------------------------------------

features = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "portfolio_drawdown",
    "VIX",
    "mkt_rf",
    "TREASURY_10Y"
]


gmm_summary = (
    df.groupby("gmm_regime")[features]
    .mean()
)

print("\nGMM Regime Characteristics:")
print(gmm_summary.to_string())

gmm_summary.to_csv(
    f"{OUTPUT_DIR}/gmm_characteristics.csv"
)


# ---------------------------------------------------------
# 4. HMM state characteristics
# ---------------------------------------------------------

hmm_summary = (
    df.groupby("hmm_regime")[features]
    .mean()
)

print("\nHMM State Characteristics:")
print(hmm_summary.to_string())

hmm_summary.to_csv(
    f"{OUTPUT_DIR}/hmm_characteristics.csv"
)


# ---------------------------------------------------------
# 5. GMM vs HMM cross-tabulation
# ---------------------------------------------------------
# IMPORTANT:
# Regime numbers are arbitrary.
# GMM regime 0 does NOT necessarily mean
# HMM state 0.
# ---------------------------------------------------------

comparison = pd.crosstab(
    df["gmm_regime"],
    df["hmm_regime"]
)

print("\nGMM vs HMM cross-tabulation:")
print(comparison)

comparison.to_csv(
    f"{OUTPUT_DIR}/gmm_hmm_crosstab.csv"
)


# ---------------------------------------------------------
# 6. Plot GMM regimes over time
# ---------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.scatter(
    df["trading_date"],
    df["gmm_regime"],
    s=8
)

plt.title("GMM Market Regimes Over Time")
plt.xlabel("Date")
plt.ylabel("GMM Regime")
plt.yticks(
    sorted(df["gmm_regime"].unique())
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/gmm_regimes.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 7. Plot HMM states over time
# ---------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.scatter(
    df["trading_date"],
    df["hmm_regime"],
    s=8
)

plt.title("HMM Market States Over Time")
plt.xlabel("Date")
plt.ylabel("HMM State")
plt.yticks(
    sorted(df["hmm_regime"].unique())
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/hmm_regimes.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 8. Plot volatility with HMM states
# ---------------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    df["trading_date"],
    df["portfolio_volatility_20d"]
)

plt.title(
    "Portfolio Volatility and HMM Regime Timeline"
)

plt.xlabel("Date")
plt.ylabel("20-Day Portfolio Volatility")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/volatility_timeline.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 9. Save final regime analysis dataset
# ---------------------------------------------------------

df.to_csv(
    f"{OUTPUT_DIR}/regime_analysis_dataset.csv",
    index=False
)


print("\n--------------------------------")
print("REGIME ANALYSIS COMPLETE")
print("--------------------------------")

print("\nFiles saved to:")
print(OUTPUT_DIR)
import pandas as pd
import os


GMM_FILE = "data/processed/regime/gmm/all_regimes.csv"
HMM_FILE = "data/processed/regime/hmm/all_regimes.csv"

OUTPUT_DIR = "data/processed/regime"


# ---------------------------------------------------------
# 1. Load GMM and HMM results
# ---------------------------------------------------------

gmm = pd.read_csv(GMM_FILE)
hmm = pd.read_csv(HMM_FILE)

gmm["trading_date"] = pd.to_datetime(
    gmm["trading_date"]
)

hmm["trading_date"] = pd.to_datetime(
    hmm["trading_date"]
)


print("GMM rows:", len(gmm))
print("HMM rows:", len(hmm))


# ---------------------------------------------------------
# 2. Select GMM regime information
# ---------------------------------------------------------

gmm_columns = [
    "trading_date",
    "regime",
    "regime_0_probability",
    "regime_1_probability",
    "regime_2_probability",
    "regime_3_probability"
]

gmm_selected = gmm[gmm_columns].copy()

gmm_selected = gmm_selected.rename(
    columns={
        "regime": "gmm_regime",
        "regime_0_probability": "gmm_regime_0_probability",
        "regime_1_probability": "gmm_regime_1_probability",
        "regime_2_probability": "gmm_regime_2_probability",
        "regime_3_probability": "gmm_regime_3_probability"
    }
)


# ---------------------------------------------------------
# 3. Select HMM regime information
# ---------------------------------------------------------

hmm_columns = [
    "trading_date",
    "hmm_regime",
    "hmm_state_0_probability",
    "hmm_state_1_probability",
    "hmm_state_2_probability",
    "hmm_state_3_probability"
]

hmm_selected = hmm[hmm_columns].copy()


# ---------------------------------------------------------
# 4. Merge GMM + HMM
# ---------------------------------------------------------

combined = gmm_selected.merge(
    hmm_selected,
    on="trading_date",
    how="inner"
)


# ---------------------------------------------------------
# 5. Add market information
# ---------------------------------------------------------

market_columns = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "portfolio_drawdown",
    "VIX",
    "mkt_rf",
    "TREASURY_10Y"
]

market_data = hmm[
    ["trading_date"] + market_columns
].copy()

market_data = market_data.drop_duplicates(
    "trading_date"
)


combined = combined.merge(
    market_data,
    on="trading_date",
    how="left"
)


# ---------------------------------------------------------
# 6. Sort chronologically
# ---------------------------------------------------------

combined = (
    combined
    .sort_values("trading_date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

output_file = (
    f"{OUTPUT_DIR}/combined_regimes.csv"
)

combined.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------

print("\n--------------------------------")
print("GMM + HMM COMBINATION COMPLETE")
print("--------------------------------")

print("\nShape:")
print(combined.shape)

print("\nDate range:")
print(
    combined["trading_date"].min(),
    "to",
    combined["trading_date"].max()
)

print("\nGMM regime counts:")
print(
    combined["gmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nHMM state counts:")
print(
    combined["hmm_regime"]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")
print(combined.isnull().sum())

print("\nSaved to:")
print(output_file)
import pandas as pd


INPUT_FILE = "data/processed/regime/combined_regimes.csv"

OUTPUT_FILE = "data/processed/regime/combined_regimes_labeled.csv"

MAPPING_FILE = "data/processed/regime/regime_label_mapping.csv"


# ---------------------------------------------------------
# 1. Load combined GMM + HMM dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["trading_date"] = pd.to_datetime(
    df["trading_date"]
)


# ---------------------------------------------------------
# 2. Define semantic labels
# ---------------------------------------------------------
#
# IMPORTANT:
# These are descriptive interpretations of the
# discovered states. They are NOT ground-truth labels.
# ---------------------------------------------------------

gmm_labels = {
    0: "Calm / Low Volatility",
    1: "Favorable / Low Drawdown",
    2: "Stressed / Elevated Volatility",
    3: "Extreme Volatility / Severe Stress"
}


hmm_labels = {
    0: "Favorable / Low Drawdown",
    1: "Stressed / Elevated Volatility",
    2: "Extreme Volatility / Severe Stress",
    3: "Calm / Low Volatility"
}


# ---------------------------------------------------------
# 3. Add labels
# ---------------------------------------------------------

df["gmm_label"] = df["gmm_regime"].map(
    gmm_labels
)

df["hmm_label"] = df["hmm_regime"].map(
    hmm_labels
)


# ---------------------------------------------------------
# 4. Create explicit mapping table
# ---------------------------------------------------------

mapping = pd.DataFrame({
    "method": [
        "GMM",
        "GMM",
        "GMM",
        "GMM",
        "HMM",
        "HMM",
        "HMM",
        "HMM"
    ],

    "state_id": [
        0, 1, 2, 3,
        0, 1, 2, 3
    ],

    "semantic_label": [
        gmm_labels[0],
        gmm_labels[1],
        gmm_labels[2],
        gmm_labels[3],
        hmm_labels[0],
        hmm_labels[1],
        hmm_labels[2],
        hmm_labels[3]
    ]
})


# ---------------------------------------------------------
# 5. Save labeled dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 6. Save mapping
# ---------------------------------------------------------

mapping.to_csv(
    MAPPING_FILE,
    index=False
)


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

print("--------------------------------")
print("REGIME LABELING COMPLETE")
print("--------------------------------")

print("\nGMM labels:")
for state, label in gmm_labels.items():
    print(f"GMM {state} -> {label}")

print("\nHMM labels:")
for state, label in hmm_labels.items():
    print(f"HMM {state} -> {label}")

print("\nGMM label counts:")
print(
    df["gmm_label"]
    .value_counts()
)

print("\nHMM label counts:")
print(
    df["hmm_label"]
    .value_counts()
)

print("\nSaved labeled dataset:")
print(OUTPUT_FILE)

print("\nSaved mapping:")
print(MAPPING_FILE)
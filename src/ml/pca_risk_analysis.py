
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "forecasting" / "forecasting_dataset.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pca"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

exclude_columns = [
    "trading_date",
    "high_future_risk",
    "future_volatility_10d"
]

features = df.drop(
    columns=[col for col in exclude_columns if col in df.columns]
)

features = features.select_dtypes(include=np.number)

print("\nFeatures used:")
print(features.columns.tolist())

# --------------------------------------------------
# 4. STANDARDIZE FEATURES
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# --------------------------------------------------
# 5. APPLY PCA
# --------------------------------------------------

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# 6. EXPLAINED VARIANCE
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

variance_df = pd.DataFrame({
    "Principal Component": [
        f"PC{i+1}" for i in range(len(explained_variance))
    ],
    "Explained Variance": explained_variance,
    "Cumulative Variance": cumulative_variance
})

variance_df.to_csv(
    OUTPUT_DIR / "explained_variance.csv",
    index=False
)

print("\nExplained Variance:")
print(variance_df)

# --------------------------------------------------
# 7. NUMBER OF COMPONENTS FOR 95% VARIANCE
# --------------------------------------------------

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(
    f"\nComponents needed to explain 95% variance: "
    f"{n_components_95}"
)

# --------------------------------------------------
# 8. PCA LOADINGS
# --------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    index=features.columns,
    columns=[
        f"PC{i+1}" for i in range(len(features.columns))
    ]
)

loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")

print("\nPCA Loadings:")
print(loadings)

# --------------------------------------------------
# 9. SAVE REDUCED DATASET
# --------------------------------------------------

reduced_df = pd.DataFrame(
    X_pca[:, :n_components_95],
    columns=[
        f"PC{i+1}" for i in range(n_components_95)
    ]
)

reduced_df.insert(
    0,
    "trading_date",
    df["trading_date"]
)

reduced_df.to_csv(
    OUTPUT_DIR / "pca_reduced_dataset.csv",
    index=False
)

# --------------------------------------------------
# 10. EXPLAINED VARIANCE PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "explained_variance_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPCA analysis completed successfully!")
print("Results saved to:", OUTPUT_DIR)
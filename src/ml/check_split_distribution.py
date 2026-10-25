
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "forecasting"

for split in ["train", "validation", "test"]:

    path = DATA_DIR / f"y_{split}.csv"

    y = pd.read_csv(path).squeeze()

    print(f"\n{split.upper()} SET")
    print("-" * 35)

    print("Total observations:", len(y))
    print("\nClass counts:")
    print(y.value_counts(dropna=False).sort_index())

    print("\nClass proportions:")
    print(y.value_counts(normalize=True, dropna=False).sort_index())

# Check original target distribution
dataset = pd.read_csv(DATA_DIR / "forecasting_dataset.csv")

print("\nFULL DATASET")
print("-" * 35)

print(dataset["high_future_risk"].value_counts())
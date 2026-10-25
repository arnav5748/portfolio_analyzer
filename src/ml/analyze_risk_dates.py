
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "forecasting"

df = pd.read_csv(DATA_DIR / "forecasting_dataset.csv")

df["trading_date"] = pd.to_datetime(df["trading_date"])

# Reconstruct the original chronological split boundaries
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df["split"] = "test"
df.loc[:train_end - 1, "split"] = "train"
df.loc[train_end:val_end - 1, "split"] = "validation"

print("\nDATE RANGES")
print("-" * 50)

for split in ["train", "validation", "test"]:
    part = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("Start:", part["trading_date"].min())
    print("End:", part["trading_date"].max())
    print("High-risk events:", (part["high_future_risk"] == 1).sum())
    print("Total rows:", len(part))

print("\nHIGH-RISK OBSERVATIONS BY YEAR")
print("-" * 50)

positive = df[df["high_future_risk"] == 1].copy()

positive["year"] = positive["trading_date"].dt.year

print(
    positive.groupby("year").size()
)

print("\nHIGH-RISK DATES")
print("-" * 50)

print(
    positive[
        ["trading_date"]
    ].to_string(index=False)
)
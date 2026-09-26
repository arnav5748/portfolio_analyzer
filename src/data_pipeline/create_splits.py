
import os
import pandas as pd


INPUT_PATH = "data/processed/portfolio_ml_dataset.csv"
OUTPUT_DIR = "data/processed/splits"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["trading_date"] = pd.to_datetime(
        data["trading_date"]
    )

    data = data.sort_values(
        "trading_date"
    ).reset_index(drop=True)

    print("Original dataset:", data.shape)

    # Remove rows without a valid target
    data = data.dropna(
        subset=["high_future_risk"]
    ).copy()

    # Remove rows without the minimum required
    # current-time portfolio features
    required_features = [
        "portfolio_return",
        "portfolio_volatility_20d",
        "annualized_portfolio_volatility",
        "portfolio_drawdown"
    ]

    data = data.dropna(
        subset=required_features
    ).copy()

    data["high_future_risk"] = (
        data["high_future_risk"].astype(int)
    )

    data = data.sort_values(
        "trading_date"
    ).reset_index(drop=True)

    print("After cleaning:", data.shape)

    # -----------------------------------
    # CHRONOLOGICAL SPLIT
    # -----------------------------------

    total_rows = len(data)

    train_end = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)

    train = data.iloc[:train_end].copy()

    validation = data.iloc[
        train_end:validation_end
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    # -----------------------------------
    # SAVE FILES
    # -----------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train.to_csv(
        f"{OUTPUT_DIR}/train.csv",
        index=False
    )

    validation.to_csv(
        f"{OUTPUT_DIR}/validation.csv",
        index=False
    )

    test.to_csv(
        f"{OUTPUT_DIR}/test.csv",
        index=False
    )

    # -----------------------------------
    # VALIDATION
    # -----------------------------------

    print("\n========== SPLIT SUMMARY ==========")

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:
        print(f"\n{name}")
        print("Rows:", len(split))

        print(
            "Date range:",
            split["trading_date"].min(),
            "to",
            split["trading_date"].max()
        )

        print("Target distribution:")
        print(
            split["high_future_risk"]
            .value_counts()
            .sort_index()
        )

    # Check chronological order
    assert (
        train["trading_date"].max()
        < validation["trading_date"].min()
    )

    assert (
        validation["trading_date"].max()
        < test["trading_date"].min()
    )

    # Check that all target values are valid
    assert set(
        train["high_future_risk"].unique()
    ).issubset({0, 1})

    assert set(
        validation["high_future_risk"].unique()
    ).issubset({0, 1})

    assert set(
        test["high_future_risk"].unique()
    ).issubset({0, 1})

    print("\nChronological checks passed.")
    print("Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
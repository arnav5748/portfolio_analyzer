
import os
import pandas as pd


INPUT_PATH = (
    "data/processed/portfolio_ml_dataset.csv"
)


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["trading_date"] = pd.to_datetime(
        data["trading_date"]
    )

    print("=" * 60)
    print("PORTFOLIO ML DATASET AUDIT")
    print("=" * 60)

    # -----------------------------------
    # 1. DATASET OVERVIEW
    # -----------------------------------

    print("\n========== DATASET OVERVIEW ==========")

    print("Shape:", data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nDate range:")
    print("Start:", data["trading_date"].min())
    print("End:", data["trading_date"].max())

    # -----------------------------------
    # 2. MISSING VALUES
    # -----------------------------------

    print("\n========== MISSING VALUES ==========")

    missing = data.isnull().sum()

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    # -----------------------------------
    # 3. TARGET DISTRIBUTION
    # -----------------------------------

    print("\n========== TARGET DISTRIBUTION ==========")

    print(
        data["high_future_risk"]
        .value_counts(dropna=False)
    )

    print("\nTarget percentages:")

    print(
        data["high_future_risk"]
        .value_counts(
            normalize=True,
            dropna=False
        ).mul(100).round(2)
    )

    # -----------------------------------
    # 4. PORTFOLIO RETURN STATISTICS
    # -----------------------------------

    print(
        "\n========== PORTFOLIO STATISTICS =========="
    )

    portfolio_columns = [
        "portfolio_return",
        "portfolio_volatility_20d",
        "annualized_portfolio_volatility",
        "portfolio_drawdown",
        "future_volatility_10d"
    ]

    print(
        data[portfolio_columns].describe()
    )

    # -----------------------------------
    # 5. NUMBER OF ASSETS
    # -----------------------------------

    print(
        "\n========== ASSET COUNT =========="
    )

    print(
        data["number_of_assets"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------
    # 6. DUPLICATES
    # -----------------------------------

    print("\n========== DUPLICATES ==========")

    print(
        "Duplicate dates:",
        data["trading_date"].duplicated().sum()
    )

    # -----------------------------------
    # 7. LAST RECORDS
    # -----------------------------------

    print("\n========== LAST 10 RECORDS ==========")

    print(data.tail(10))

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()
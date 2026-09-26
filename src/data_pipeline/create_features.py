import os
import pandas as pd


INPUT_PATH = "data/processed/cleaned_stock_prices.csv"
OUTPUT_PATH = "data/processed/stock_features.csv"


def load_cleaned_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["trading_date"] = pd.to_datetime(
        data["trading_date"]
    )

    return data



def calculate_features(data):
    data = data.copy()

    # Sort by asset and date
    data = data.sort_values(
        by=["asset_id", "trading_date"]
    ).reset_index(drop=True)

    # -----------------------------------
    # 1. DAILY RETURNS
    # -----------------------------------

    data["daily_return"] = (
        data.groupby("asset_id")["adjusted_close"]
        .pct_change(fill_method=None)
    )

    # -----------------------------------
    # 2. ROLLING VOLATILITY
    # -----------------------------------

    grouped_returns = data.groupby("asset_id")["daily_return"]

    data["rolling_volatility_20d"] = (
        grouped_returns.transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).std()
        )
    )

    data["annualized_volatility_20d"] = (
        data["rolling_volatility_20d"] * (252 ** 0.5)
    )

    # -----------------------------------
    # 3. MOMENTUM
    # -----------------------------------

    grouped_prices = data.groupby("asset_id")["adjusted_close"]

    data["momentum_5d"] = (
        grouped_prices.pct_change(
            periods=5,
            fill_method=None
        )
    )

    data["momentum_20d"] = (
        grouped_prices.pct_change(
            periods=20,
            fill_method=None
        )
    )

    # -----------------------------------
    # 4. MOVING AVERAGES
    # -----------------------------------

    data["moving_average_20d"] = (
        grouped_prices.transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )

    data["moving_average_50d"] = (
        grouped_prices.transform(
            lambda x: x.rolling(
                window=50,
                min_periods=50
            ).mean()
        )
    )

    # -----------------------------------
    # 5. PRICE-TO-MA RATIOS
    # -----------------------------------

    data["price_to_ma20"] = (
        data["adjusted_close"] /
        data["moving_average_20d"]
    )

    data["price_to_ma50"] = (
        data["adjusted_close"] /
        data["moving_average_50d"]
    )

    # -----------------------------------
    # 6. ROLLING MAXIMUM DRAWDOWN
    # -----------------------------------

    def rolling_max_drawdown(prices):
        rolling_peak = prices.rolling(
            window=60,
            min_periods=60
        ).max()

        drawdown = (prices / rolling_peak) - 1

        return drawdown.rolling(
            window=60,
            min_periods=60
        ).min()

    data["rolling_max_drawdown_60d"] = (
        data.groupby("asset_id")["adjusted_close"]
        .transform(rolling_max_drawdown)
    )

    # -----------------------------------
    # 7. VOLUME FEATURES
    # -----------------------------------

    grouped_volume = data.groupby("asset_id")["volume"]

    data["average_volume_20d"] = (
        grouped_volume.transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )

    data["volume_ratio_20d"] = (
        data["volume"] /
        data["average_volume_20d"]
    )

    return data


def save_features(data):
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nFeatures saved to: {OUTPUT_PATH}")


def main():
    print("Loading cleaned stock price data...")

    data = load_cleaned_data()

    print("Original shape:", data.shape)

    data = calculate_features(data)

    print("\nFeature engineering completed.")

    selected_columns = [
        "ticker",
        "trading_date",
        "adjusted_close",
        "daily_return",
        "rolling_volatility_20d",
        "annualized_volatility_20d",
        "momentum_5d",
        "momentum_20d",
        "moving_average_20d",
        "moving_average_50d",
        "price_to_ma20",
        "price_to_ma50",
        "rolling_max_drawdown_60d",
        "average_volume_20d",
        "volume_ratio_20d"
    ]

    print("\nFeature preview:")
    print(data[selected_columns].tail(10))

    print("\nFeature missing values:")
    print(data[selected_columns].isnull().sum())

    print("\nFeature statistics:")
    print(data[selected_columns].describe())

    save_features(data)


if __name__ == "__main__":
    main()
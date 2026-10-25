import pandas as pd
import numpy as np


DATA_PATH = "data/processed/cleaned_stock_prices.csv"
FEATURES_PATH = "data/processed/integrated_features.csv"


def build_custom_portfolio_features(weights):
    """
    Generate ML features for a custom portfolio.

    weights example:
    {
        "AAPL": 0.60,
        "MSFT": 0.40
    }
    """

    if not weights:
        raise ValueError("Portfolio cannot be empty.")

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    # Normalize ticker names
    weights = {
        ticker.upper(): weight
        for ticker, weight in weights.items()
    }

    # Load prices
    prices = pd.read_csv(DATA_PATH)
    prices["trading_date"] = pd.to_datetime(prices["trading_date"])

    # Check whether all selected assets exist
    available_tickers = set(prices["ticker"].unique())

    missing = set(weights.keys()) - available_tickers

    if missing:
        raise ValueError(f"Unknown tickers: {missing}")

    # Keep only selected assets
    prices = prices[prices["ticker"].isin(weights.keys())]

    # Pivot adjusted prices
    price_matrix = prices.pivot(
        index="trading_date",
        columns="ticker",
        values="adjusted_close"
    )

    # Calculate daily returns
    returns = price_matrix.pct_change()

    # Weighted portfolio return
    portfolio_returns = sum(
        returns[ticker] * weight
        for ticker, weight in weights.items()
    )

    df = pd.DataFrame({
        "trading_date": portfolio_returns.index,
        "portfolio_return": portfolio_returns.values
    })

    # Remove dates where any selected asset has no return
    df = df.dropna(subset=["portfolio_return"]).copy()

    # Rolling volatility
    df["portfolio_volatility_20d"] = (
        df["portfolio_return"]
        .rolling(window=20)
        .std()
    )

    # Annualized volatility
    df["annualized_portfolio_volatility"] = (
        df["portfolio_volatility_20d"] * np.sqrt(252)
    )

    # Portfolio growth
    df["cumulative_growth"] = (
        1 + df["portfolio_return"]
    ).cumprod()

    # Running peak
    df["running_peak"] = df["cumulative_growth"].cummax()

    # Drawdown
    df["portfolio_drawdown"] = (
        df["cumulative_growth"] / df["running_peak"] - 1
    )

    # Load market factors and macroeconomic indicators
    integrated = pd.read_csv(FEATURES_PATH)
    integrated["trading_date"] = pd.to_datetime(
        integrated["trading_date"]
    )

    exogenous_columns = [
        "trading_date",
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "rf",
        "CPI",
        "FEDERAL_FUNDS_RATE",
        "TREASURY_10Y",
        "UNEMPLOYMENT_RATE",
        "VIX"
    ]

    exogenous = (
        integrated[exogenous_columns]
        .drop_duplicates(subset=["trading_date"])
        .sort_values("trading_date")
        .reset_index(drop=True)
    )

    # Carry forward the latest available factor and macro values
    factor_columns = [
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "rf",
        "CPI",
        "FEDERAL_FUNDS_RATE",
        "TREASURY_10Y",
        "UNEMPLOYMENT_RATE",
        "VIX",
    ]

    exogenous[factor_columns] = (
        exogenous[factor_columns].ffill()
    )

    # Merge by date
    df = df.merge(
        exogenous,
        on="trading_date",
        how="left"
    )

    return df
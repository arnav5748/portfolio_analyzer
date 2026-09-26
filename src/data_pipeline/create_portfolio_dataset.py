import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/integrated_features.csv"
OUTPUT_FILE = "data/processed/portfolio_ml_dataset.csv"


# ---------------------------------------------------------
# 1. Load integrated dataset
# ---------------------------------------------------------

data = pd.read_csv(INPUT_FILE)

data["trading_date"] = pd.to_datetime(data["trading_date"])

print("Integrated dataset loaded.")
print("Rows:", len(data))
print("Columns:", len(data.columns))


# ---------------------------------------------------------
# 2. Portfolio-level return
# ---------------------------------------------------------
# Equal-weight portfolio:
# Every available asset gets equal weight on each date.
#
# portfolio_return = mean(asset daily returns)
# ---------------------------------------------------------

portfolio = (
    data.groupby("trading_date")
    .agg(
        portfolio_return=("daily_return", "mean"),
        number_of_assets=("asset_id", "nunique")
    )
    .reset_index()
)


# ---------------------------------------------------------
# 3. Portfolio volatility
# ---------------------------------------------------------

portfolio["portfolio_volatility_20d"] = (
    portfolio["portfolio_return"]
    .rolling(window=20)
    .std()
)

portfolio["annualized_portfolio_volatility"] = (
    portfolio["portfolio_volatility_20d"] * np.sqrt(252)
)


# ---------------------------------------------------------
# 4. Portfolio cumulative growth
# ---------------------------------------------------------

portfolio["cumulative_growth"] = (
    (1 + portfolio["portfolio_return"]).cumprod()
)


# ---------------------------------------------------------
# 5. Running peak
# ---------------------------------------------------------

portfolio["running_peak"] = (
    portfolio["cumulative_growth"].cummax()
)


# ---------------------------------------------------------
# 6. Portfolio drawdown
# ---------------------------------------------------------

portfolio["portfolio_drawdown"] = (
    portfolio["cumulative_growth"] / portfolio["running_peak"] - 1
)


# ---------------------------------------------------------
# 7. Future 10-day volatility
# ---------------------------------------------------------
# IMPORTANT:
# This uses the NEXT 10 trading days.
# It is used only to create the target.
# It must NOT be used as an ML input feature.
# ---------------------------------------------------------

future_returns = portfolio["portfolio_return"].shift(-1)

portfolio["future_volatility_10d"] = (
    future_returns
    .iloc[::-1]
    .rolling(window=10, min_periods=10)
    .std()
    .iloc[::-1]
    .values
)


# ---------------------------------------------------------
# 8. Create high-risk target
# ---------------------------------------------------------
# High risk = future 10-day volatility >= historical
# 90th percentile.
#
# Last 10 rows do not have enough future data, so their
# target remains NaN.
# ---------------------------------------------------------

valid_future_volatility = portfolio["future_volatility_10d"].dropna()

risk_threshold = valid_future_volatility.quantile(0.90)

portfolio["high_future_risk"] = pd.Series(
    pd.NA,
    index=portfolio.index,
    dtype="Int64"
)

valid_mask = portfolio["future_volatility_10d"].notna()

portfolio.loc[valid_mask, "high_future_risk"] = (
    portfolio.loc[valid_mask, "future_volatility_10d"]
    >= risk_threshold
).astype("int64")


# ---------------------------------------------------------
# 9. Bring Fama-French + Macro variables to portfolio level
# ---------------------------------------------------------
#
# These variables are DATE-LEVEL variables.
#
# They are not averaged across stocks.
# There should be one value for each trading date.
# ---------------------------------------------------------

market_columns = [
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


market_data = (
    data.groupby("trading_date")[market_columns]
    .first()
    .reset_index()
)


# ---------------------------------------------------------
# 10. Merge market data with portfolio data
# ---------------------------------------------------------

portfolio = portfolio.merge(
    market_data,
    on="trading_date",
    how="left"
)


# ---------------------------------------------------------
# 11. Arrange columns
# ---------------------------------------------------------

portfolio = portfolio[
    [
        "trading_date",

        # Portfolio features
        "portfolio_return",
        "number_of_assets",
        "portfolio_volatility_20d",
        "annualized_portfolio_volatility",
        "cumulative_growth",
        "running_peak",
        "portfolio_drawdown",

        # Fama-French factors
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "rf",

        # Macro indicators
        "CPI",
        "FEDERAL_FUNDS_RATE",
        "TREASURY_10Y",
        "UNEMPLOYMENT_RATE",
        "VIX",

        # Future information / target
        "future_volatility_10d",
        "high_future_risk"
    ]
]


# ---------------------------------------------------------
# 12. Save dataset
# ---------------------------------------------------------

portfolio.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# 13. Print summary
# ---------------------------------------------------------

print("\nPortfolio dataset created successfully.")

print("\nShape:")
print(portfolio.shape)

print("\nDate range:")
print(portfolio["trading_date"].min(), "to", portfolio["trading_date"].max())

print("\nColumns:")
print(portfolio.columns.tolist())

print("\nMissing values:")
print(portfolio.isnull().sum())

print("\nRisk threshold:")
print(risk_threshold)

print("\nRisk label distribution:")
print(portfolio["high_future_risk"].value_counts(dropna=False))

print("\nNumber of assets:")
print(portfolio["number_of_assets"].describe())

print("\nSaved to:")
print(OUTPUT_FILE)

import pandas as pd
import numpy as np


DATA_PATH = "data/processed/cleaned_stock_prices.csv"


def analyze_portfolio(weights, data_path=DATA_PATH):
    """
    Analyze a custom portfolio.

    Parameters
    ----------
    weights : dict
        Dictionary containing ticker and portfolio weight.
        Example: {"AAPL": 0.4, "MSFT": 0.35, "NVDA": 0.25}

    Returns
    -------
    dict
        Portfolio performance, risk metrics, and asset correlations.
    """

    # 1. Validate portfolio weights
    if not weights:
        raise ValueError("Portfolio cannot be empty.")

    weights = {ticker.upper(): float(weight) for ticker, weight in weights.items()}

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Negative weights are not supported.")

    total_weight = sum(weights.values())

    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError("Portfolio weights must sum to 1.")

    # 2. Load historical stock prices
    df = pd.read_csv(data_path, parse_dates=["trading_date"])

    selected_tickers = list(weights.keys())

    available_tickers = set(df["ticker"].unique())
    invalid_tickers = set(selected_tickers) - available_tickers

    if invalid_tickers:
        raise ValueError(
            f"Unknown tickers: {sorted(invalid_tickers)}"
        )

    # 3. Prepare adjusted closing prices
    prices = df[
        df["ticker"].isin(selected_tickers)
    ].pivot(
        index="trading_date",
        columns="ticker",
        values="adjusted_close"
    )

    prices = prices.sort_index()

    # Use dates available for every selected asset
    prices = prices.dropna()

    if len(prices) < 2:
        raise ValueError("Insufficient overlapping price history.")

    # 4. Calculate daily asset returns
    asset_returns = prices.pct_change().dropna()

    # 5. Calculate weighted portfolio returns
    weight_series = pd.Series(weights)

    portfolio_returns = (
        asset_returns * weight_series
    ).sum(axis=1)

    # 6. Calculate cumulative portfolio growth
    cumulative_growth = (1 + portfolio_returns).cumprod()

    # 7. Calculate drawdown
    running_peak = cumulative_growth.cummax()

    drawdown = (
        cumulative_growth / running_peak
    ) - 1

    # 8. Annualized performance and risk
    trading_days = 252

    total_return = cumulative_growth.iloc[-1] - 1

    years = len(portfolio_returns) / trading_days

    annualized_return = (
        cumulative_growth.iloc[-1] ** (1 / years) - 1
        if years > 0 else 0
    )

    annualized_volatility = (
        portfolio_returns.std() * np.sqrt(trading_days)
    )

    # 9. Sharpe ratio (using zero risk-free rate initially)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0 else 0
    )

    # 10. Historical Value at Risk
    var_95 = portfolio_returns.quantile(0.05)

    # 11. Expected Shortfall
    worst_returns = portfolio_returns[
        portfolio_returns <= var_95
    ]

    expected_shortfall_95 = worst_returns.mean()

    # 12. Maximum drawdown
    max_drawdown = drawdown.min()

    # 13. Portfolio beta against SPY benchmark
    beta = None

    spy_prices = df[df["ticker"] == "SPY"].set_index("trading_date")[
        "adjusted_close"
    ].sort_index()

    spy_returns = spy_prices.pct_change().dropna()

    aligned_returns = pd.concat(
        [portfolio_returns, spy_returns.rename("SPY")],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned_returns) > 1:
        market_variance = aligned_returns["SPY"].var()

        if market_variance != 0:
            beta = (
                aligned_returns.iloc[:, 0].cov(aligned_returns["SPY"])
                / market_variance
            )

    # 14. Correlation matrix
    correlation_matrix = asset_returns.corr()

    # 15. Prepare output
    results = {
        "portfolio": weights,
        "start_date": str(portfolio_returns.index[0].date()),
        "end_date": str(portfolio_returns.index[-1].date()),
        "observations": len(portfolio_returns),

        "performance": {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "cumulative_growth": float(cumulative_growth.iloc[-1])
        },

        "risk": {
            "annualized_volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "expected_shortfall_95": float(expected_shortfall_95),
            "beta": float(beta) if beta is not None else None
        },

        "correlation_matrix": correlation_matrix.to_dict(),
    }

    return results


if __name__ == "__main__":

    sample_portfolio = {
        "AAPL": 0.40,
        "MSFT": 0.35,
        "NVDA": 0.25
    }

    result = analyze_portfolio(sample_portfolio)

    print("\nPORTFOLIO PERFORMANCE")
    for key, value in result["performance"].items():
        print(f"{key}: {value:.4f}")

    print("\nPORTFOLIO RISK")
    for key, value in result["risk"].items():
        print(f"{key}: {value}")

    print("\nCORRELATION MATRIX")
    print(pd.DataFrame(result["correlation_matrix"]))
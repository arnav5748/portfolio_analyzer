
from src.ml.custom_portfolio_features import (
    build_custom_portfolio_features
)

from src.ml.inference_engine import analyze_custom_portfolio


portfolio = {
    "AAPL": 0.60,
    "MSFT": 0.40,
}

features = build_custom_portfolio_features(portfolio)

results = analyze_custom_portfolio(features)

print("\nCUSTOM PORTFOLIO ML RESULTS")
print("=" * 45)

print(results)
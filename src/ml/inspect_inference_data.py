import pandas as pd

files = {
    "Cleaned Stock Prices": "data/processed/cleaned_stock_prices.csv",
    "Integrated Features": "data/processed/integrated_features.csv",
    "Portfolio Dataset": "data/processed/portfolio_ml_dataset.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    date_col = next(
        (col for col in df.columns if "date" in col.lower()),
        None
    )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        print("Date range:", df[date_col].min(), "to", df[date_col].max())

    print("\nSample:")
    print(df.head(2).to_string(index=False))
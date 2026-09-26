
import os
from urllib.parse import urlparse, unquote

import mysql.connector
import pandas as pd
from dotenv import load_dotenv


def create_database_connection():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file")

    url = urlparse(database_url)

    connection = mysql.connector.connect(
        host=url.hostname,
        port=url.port or 3306,
        user=unquote(url.username),
        password=unquote(url.password),
        database=url.path.lstrip("/")
    )

    return connection


def extract_stock_prices(connection):
    query = """
        SELECT
            sp.asset_id,
            a.ticker,
            sp.trading_date,
            sp.open_price,
            sp.high_price,
            sp.low_price,
            sp.close_price,
            sp.adjusted_close,
            sp.volume
        FROM stock_prices sp
        JOIN assets a
            ON sp.asset_id = a.asset_id;
    """

    stock_prices = pd.read_sql(query, connection)

    return stock_prices


def clean_stock_prices(stock_prices):
    data = stock_prices.copy()

    # Convert date column
    data["trading_date"] = pd.to_datetime(
        data["trading_date"]
    )

    # Convert price columns to numeric
    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close"
    ]

    for column in price_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Convert volume to numeric
    data["volume"] = pd.to_numeric(
        data["volume"],
        errors="coerce"
    )

    # Remove duplicate asset-date records
    data = data.drop_duplicates(
        subset=["asset_id", "trading_date"]
    )

    # Sort by asset and trading date
    data = data.sort_values(
        by=["asset_id", "trading_date"]
    ).reset_index(drop=True)

    # Remove rows with missing essential values
    essential_columns = [
        "asset_id",
        "ticker",
        "trading_date",
        "adjusted_close"
    ]

    data = data.dropna(
        subset=essential_columns
    )

    return data


def main():
    connection = None

    try:
        connection = create_database_connection()

        print("Database connection successful.")

        stock_prices = extract_stock_prices(connection)

        print("\nBefore cleaning:", stock_prices.shape)

        cleaned_data = clean_stock_prices(stock_prices)

        print("After cleaning:", cleaned_data.shape)

        print("\nCleaned data preview:")
        print(cleaned_data.head())

        print("\nMissing values after cleaning:")
        print(cleaned_data.isnull().sum())

        # Save the cleaned dataset
        os.makedirs("data/processed", exist_ok=True)

        output_path = "data/processed/cleaned_stock_prices.csv"

        cleaned_data.to_csv(
            output_path,
            index=False
        )

        print(f"\nCleaned data saved to: {output_path}")

    except Exception as error:
        print("Cleaning error:", error)

    finally:
        if connection is not None and connection.is_connected():
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()

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


def load_stock_prices(connection):
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

    return pd.read_sql(query, connection)


def validate_stock_prices(stock_prices):
    print("\n========== DATASET OVERVIEW ==========")

    print("Rows:", stock_prices.shape[0])
    print("Columns:", stock_prices.shape[1])

    print("\n========== DATA TYPES ==========")
    print(stock_prices.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(stock_prices.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(stock_prices.duplicated().sum())

    print("\n========== DATE RANGE ==========")
    stock_prices["trading_date"] = pd.to_datetime(
        stock_prices["trading_date"]
    )

    print("Start date:", stock_prices["trading_date"].min())
    print("End date:", stock_prices["trading_date"].max())

    print("\n========== INVALID PRICES ==========")

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close"
    ]

    for column in price_columns:
        invalid_count = (stock_prices[column] <= 0).sum()
        print(f"{column}: {invalid_count} invalid values")

    print("\n========== INVALID VOLUME ==========")

    invalid_volume = (stock_prices["volume"] < 0).sum()

    print("Negative volume values:", invalid_volume)

    print("\n========== ASSET COVERAGE ==========")

    print(
        stock_prices["ticker"].nunique(),
        "unique assets found"
    )

    print("\n========== VALIDATION COMPLETE ==========")


def main():
    connection = None

    try:
        connection = create_database_connection()

        print("Database connection successful.")

        stock_prices = load_stock_prices(connection)

        validate_stock_prices(stock_prices)

    except Exception as error:
        print("Validation error:", error)

    finally:
        if connection is not None and connection.is_connected():
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()

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


def extract_assets(connection):
    query = """
        SELECT
            asset_id,
            ticker,
            company_name,
            sector,
            asset_type
        FROM assets
        ORDER BY asset_id;
    """

    assets = pd.read_sql(query, connection)

    return assets


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
            ON sp.asset_id = a.asset_id
        ORDER BY
            sp.trading_date,
            sp.asset_id;
    """

    stock_prices = pd.read_sql(query, connection)

    stock_prices["trading_date"] = pd.to_datetime(
        stock_prices["trading_date"]
    )

    return stock_prices

def extract_factor_returns(connection):
    query = """
        SELECT
            factor_date,
            mkt_rf,
            smb,
            hml,
            rmw,
            cma,
            mom,
            rf
        FROM factor_returns
        ORDER BY factor_date;
    """

    factors = pd.read_sql(query, connection)

    factors["factor_date"] = pd.to_datetime(
        factors["factor_date"]
    )

    return factors

def extract_macro_indicators(connection):
    query = """
        SELECT
            macro_id,
            indicator_date,
            indicator_name,
            indicator_value
        FROM macro_indicators
        ORDER BY indicator_date, indicator_name;
    """

    macro_data = pd.read_sql(query, connection)

    macro_data["indicator_date"] = pd.to_datetime(
        macro_data["indicator_date"]
    )

    return macro_data


def main():
    connection = None

    try:
        connection = create_database_connection()

        print("Database connection successful.")

        assets = extract_assets(connection)
        stock_prices = extract_stock_prices(connection)
        factors = extract_factor_returns(connection)
        macro_data = extract_macro_indicators(connection)

        print("\nAssets extracted:", assets.shape)
        print(assets.head())

        print("\nStock prices extracted:", stock_prices.shape)
        print(stock_prices.head())

        print("\nFactor returns extracted:", factors.shape)
        print(factors.head())

        print("\nMacro indicators extracted:", macro_data.shape)
        print(macro_data.head())

        print("\nExtraction completed successfully.")

    except Exception as error:
        print("Error during data extraction:", error)

    finally:
        if connection is not None and connection.is_connected():
            connection.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()
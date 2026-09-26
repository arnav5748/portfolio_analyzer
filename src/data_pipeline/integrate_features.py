
import os
from urllib.parse import urlparse, unquote

import mysql.connector
import pandas as pd
from dotenv import load_dotenv


STOCK_FEATURES_PATH = (
    "data/processed/stock_features.csv"
)

OUTPUT_PATH = (
    "data/processed/integrated_features.csv"
)


def create_database_connection():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL not found in .env file"
        )

    url = urlparse(database_url)

    connection = mysql.connector.connect(
        host=url.hostname,
        port=url.port or 3306,
        user=unquote(url.username),
        password=unquote(url.password),
        database=url.path.lstrip("/")
    )

    return connection


def load_stock_features():
    data = pd.read_csv(STOCK_FEATURES_PATH)

    data["trading_date"] = pd.to_datetime(
        data["trading_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    return data.sort_values(
        ["trading_date", "asset_id"]
    ).reset_index(drop=True)


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
        factors["factor_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    factors = factors.rename(
        columns={"factor_date": "trading_date"}
    )

    return factors



def extract_macro_indicators(connection):
    query = """
        SELECT
            indicator_date,
            indicator_name,
            indicator_value
        FROM macro_indicators
        ORDER BY indicator_date;
    """

    macro = pd.read_sql(query, connection)

    macro["indicator_date"] = pd.to_datetime(
        macro["indicator_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    # Convert long format to wide format
    macro_wide = macro.pivot_table(
        index="indicator_date",
        columns="indicator_name",
        values="indicator_value",
        aggfunc="last"
    ).reset_index()

    macro_wide = macro_wide.rename(
        columns={
            "indicator_date": "trading_date"
        }
    )

    macro_wide["trading_date"] = pd.to_datetime(
        macro_wide["trading_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    macro_wide = macro_wide.sort_values(
        "trading_date"
    ).reset_index(drop=True)

    # Forward-fill each indicator independently.
    # Uses only the latest known historical value.
    macro_columns = [
        column
        for column in macro_wide.columns
        if column != "trading_date"
    ]

    macro_wide[macro_columns] = (
        macro_wide[macro_columns].ffill()
    )

    return macro_wide


def integrate_features(
    stock_features,
    factors,
    macro_data
):
    stock_features = stock_features.copy()
    factors = factors.copy()
    macro_data = macro_data.copy()

    # Standardize all merge keys to the same datetime type
    for dataframe in [
        stock_features,
        factors,
        macro_data
    ]:
        dataframe["trading_date"] = pd.to_datetime(
            dataframe["trading_date"],
            errors="coerce"
        ).astype("datetime64[ns]")

    # Remove invalid dates
    stock_features = stock_features.dropna(
        subset=["trading_date"]
    )

    factors = factors.dropna(
        subset=["trading_date"]
    )

    macro_data = macro_data.dropna(
        subset=["trading_date"]
    )

    # Sort by date before merging
    stock_features = stock_features.sort_values(
        "trading_date"
    )

    factors = factors.sort_values(
        "trading_date"
    )

    macro_data = macro_data.sort_values(
        "trading_date"
    )

    # Merge factors on exact date
    integrated = stock_features.merge(
        factors,
        on="trading_date",
        how="left"
    )

    # Merge the latest known macro observation
    integrated = pd.merge_asof(
        integrated.sort_values("trading_date"),
        macro_data,
        on="trading_date",
        direction="backward"
    )

    # Restore asset/date ordering
    integrated = integrated.sort_values(
        ["asset_id", "trading_date"]
    ).reset_index(drop=True)

    return integrated


def main():
    connection = None

    try:
        print("Loading stock features...")

        stock_features = load_stock_features()

        connection = create_database_connection()

        print("Database connection successful.")

        factors = extract_factor_returns(connection)

        macro_data = extract_macro_indicators(connection)

        print("\nStock features shape:", stock_features.shape)
        print("Factors shape:", factors.shape)
        print("Macro data shape:", macro_data.shape)

        integrated = integrate_features(
            stock_features,
            factors,
            macro_data
        )

        print(
            "\nIntegrated dataset shape:",
            integrated.shape
        )

        print("\nIntegrated dataset preview:")
        print(integrated.head())

        print("\nMissing values:")
        print(integrated.isnull().sum())

        os.makedirs(
            "data/processed",
            exist_ok=True
        )

        integrated.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print(
            f"\nIntegrated dataset saved to: {OUTPUT_PATH}"
        )

    except Exception as error:
        print("Integration error:", error)

    finally:
        if (
            connection is not None
            and connection.is_connected()
        ):
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
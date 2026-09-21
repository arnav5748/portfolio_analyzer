
import os
from datetime import date

import requests
from dotenv import load_dotenv

from database.connection import create_connection


load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = "2020-01-01"

FRED_BASE_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


# Indicator name in our database -> FRED series ID
INDICATORS = {
    "VIX": "VIXCLS",
    "FEDERAL_FUNDS_RATE": "DFF",
    "TREASURY_10Y": "DGS10",
    "CPI": "CPIAUCSL",
    "UNEMPLOYMENT_RATE": "UNRATE",
}


def fetch_fred_series(series_id):
    """Fetch one economic series from FRED."""

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "sort_order": "asc",
    }

    response = requests.get(
        FRED_BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "error_code" in data:
        raise RuntimeError(
            f"FRED error: {data.get('error_message')}"
        )

    return data.get("observations", [])


def insert_macro_data():
    if not FRED_API_KEY:
        print("FRED_API_KEY is missing from .env")
        return

    connection = create_connection()

    if connection is None:
        print("Database connection failed.")
        return

    cursor = connection.cursor()

    insert_query = """
        INSERT IGNORE INTO macro_indicators (
            indicator_date,
            indicator_name,
            indicator_value
        )
        VALUES (%s, %s, %s);
    """

    total_inserted = 0

    try:
        for indicator_name, series_id in INDICATORS.items():

            print(
                f"\nFetching {indicator_name} "
                f"({series_id})..."
            )

            observations = fetch_fred_series(series_id)

            rows = []

            for observation in observations:

                observation_date = observation["date"]
                value = observation["value"]

                # FRED may use "." for missing observations.
                if value == ".":
                    continue

                rows.append((
                    observation_date,
                    indicator_name,
                    float(value)
                ))

            if rows:
                cursor.executemany(insert_query, rows)
                connection.commit()

                total_inserted += cursor.rowcount

                print(
                    f"{indicator_name}: "
                    f"{len(rows)} observations processed."
                )
            else:
                print(
                    f"{indicator_name}: "
                    "No valid observations found."
                )

        print(
            f"\nTotal new records inserted: "
            f"{total_inserted}"
        )

    except Exception as e:
        connection.rollback()
        print(f"Error loading macro data: {e}")

    finally:
        cursor.close()
        connection.close()
        print("MySQL connection closed.")


if __name__ == "__main__":
    insert_macro_data()
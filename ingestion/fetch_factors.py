
import io
import re
import zipfile

import pandas as pd
import requests

from database.connection import create_connection


# ============================================================
# CONFIGURATION
# ============================================================

FIVE_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

START_DATE = "2020-01-01"


# ============================================================
# DOWNLOAD ZIP FILE
# ============================================================

def download_zip_contents(url):
    """
    Download a ZIP file and return the text
    from the first file inside it.
    """

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:

        file_names = zip_file.namelist()

        if not file_names:
            raise ValueError("ZIP file is empty.")

        file_name = file_names[0]

        with zip_file.open(file_name) as file:
            content = file.read()

    return content.decode("latin1")


# ============================================================
# PARSE FAMA-FRENCH 5 FACTORS
# ============================================================

def parse_five_factors():

    print("Downloading Fama-French 5 factors...")

    text = download_zip_contents(FIVE_FACTOR_URL)

    lines = text.splitlines()

    header_index = None

    # Find the daily data header.
    for index, line in enumerate(lines):

        if "Mkt-RF" in line and "SMB" in line and "RF" in line:
            header_index = index
            break

    if header_index is None:
        raise ValueError(
            "Could not find Fama-French factor header."
        )

    rows = []

    for line in lines[header_index + 1:]:

        parts = [
            value.strip()
            for value in line.split(",")
        ]

        # Expected format:
        # Date, Mkt-RF, SMB, HML, RMW, CMA, RF
        if len(parts) < 7:
            continue

        date_text = parts[0]

        # Daily dates have YYYYMMDD format.
        if not re.fullmatch(r"\d{8}", date_text):
            continue

        try:

            factor_date = pd.to_datetime(
                date_text,
                format="%Y%m%d"
            )

            values = [
                float(value)
                for value in parts[1:7]
            ]

            rows.append({
                "factor_date": factor_date,
                "mkt_rf": values[0],
                "smb": values[1],
                "hml": values[2],
                "rmw": values[3],
                "cma": values[4],
                "rf": values[5]
            })

        except (ValueError, TypeError):
            continue

    factors_df = pd.DataFrame(rows)

    if factors_df.empty:
        raise ValueError(
            "No Fama-French 5-factor records found."
        )

    return factors_df


# ============================================================
# PARSE MOMENTUM FACTOR
# ============================================================

def parse_momentum():

    print("Downloading momentum factor...")

    text = download_zip_contents(MOMENTUM_URL)

    lines = text.splitlines()

    header_index = None

    # Find the header containing the momentum factor.
    for index, line in enumerate(lines):

        if "Mom" in line:
            header_index = index
            break

    if header_index is None:
        raise ValueError(
            "Could not find momentum factor header."
        )

    rows = []

    for line in lines[header_index + 1:]:

        parts = [
            value.strip()
            for value in line.split(",")
        ]

        if len(parts) < 2:
            continue

        date_text = parts[0]

        if not re.fullmatch(r"\d{8}", date_text):
            continue

        try:

            factor_date = pd.to_datetime(
                date_text,
                format="%Y%m%d"
            )

            momentum_value = float(parts[1])

            rows.append({
                "factor_date": factor_date,
                "mom": momentum_value
            })

        except (ValueError, TypeError):
            continue

    momentum_df = pd.DataFrame(rows)

    if momentum_df.empty:
        raise ValueError(
            "No momentum factor records found."
        )

    return momentum_df


# ============================================================
# MERGE AND PREPARE DATA
# ============================================================

def prepare_factor_data():

    five_factors = parse_five_factors()

    momentum = parse_momentum()

    print("\nMerging factor datasets...")

    factors = pd.merge(
        five_factors,
        momentum,
        on="factor_date",
        how="inner"
    )

    # Keep only data from our selected start date.
    factors = factors[
        factors["factor_date"] >= START_DATE
    ].copy()

    # Convert percentage returns into decimal returns.
    #
    # Example:
    # 1.25% becomes 0.0125
    #
    # This is important for mathematical calculations
    # such as portfolio volatility and factor exposure.

    factor_columns = [
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "rf"
    ]

    factors[factor_columns] = (
        factors[factor_columns] / 100
    )

    # Remove duplicate dates, if any.
    factors = factors.drop_duplicates(
        subset=["factor_date"]
    )

    # Remove rows containing missing values.
    factors = factors.dropna(
        subset=factor_columns
    )

    factors = factors.sort_values(
        by="factor_date"
    )

    factors = factors.reset_index(drop=True)

    return factors


# ============================================================
# INSERT INTO MYSQL
# ============================================================

def insert_factor_data():

    try:

        factors = prepare_factor_data()

    except Exception as error:

        print(
            f"Error preparing factor data: {error}"
        )

        return

    if factors.empty:

        print("No factor data available for insertion.")

        return

    print("\nFactor data preview:")
    print(factors.head())

    print(
        "\nTotal prepared factor records:",
        len(factors)
    )

    connection = create_connection()

    if connection is None:

        print("Database connection failed.")

        return

    cursor = connection.cursor()

    insert_query = """
        INSERT IGNORE INTO factor_returns (
            factor_date,
            mkt_rf,
            smb,
            hml,
            rmw,
            cma,
            mom,
            rf
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """

    rows = []

    for _, row in factors.iterrows():

        rows.append((
            row["factor_date"].date(),
            float(row["mkt_rf"]),
            float(row["smb"]),
            float(row["hml"]),
            float(row["rmw"]),
            float(row["cma"]),
            float(row["mom"]),
            float(row["rf"])
        ))

    try:

        cursor.executemany(
            insert_query,
            rows
        )

        connection.commit()

        print(
            "\nFactor records processed:",
            len(rows)
        )

        print(
            "MySQL factor data insertion completed!"
        )

    except Exception as error:

        connection.rollback()

        print(
            "Error inserting factor data:",
            error
        )

    finally:

        cursor.close()

        connection.close()

        print("MySQL connection closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    insert_factor_data()
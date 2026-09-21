
import yfinance as yf
import pandas as pd

from database.connection import create_connection


START_DATE = "2020-01-01"


def fetch_stock_prices():
    connection = create_connection()

    if connection is None:
        print("Database connection failed.")
        return

    cursor = connection.cursor(dictionary=True)

    try:
        # 1. Read tickers and their database IDs from MySQL
        cursor.execute("""
            SELECT asset_id, ticker
            FROM assets
        """)

        assets = cursor.fetchall()

        if not assets:
            print("No assets found in the database.")
            return

        print(f"Found {len(assets)} assets in MySQL.")

        insert_cursor = connection.cursor()

        insert_query = """
            INSERT INTO stock_prices (
                asset_id,
                trading_date,
                open_price,
                high_price,
                low_price,
                close_price,
                adjusted_close,
                volume
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open_price = VALUES(open_price),
                high_price = VALUES(high_price),
                low_price = VALUES(low_price),
                close_price = VALUES(close_price),
                adjusted_close = VALUES(adjusted_close),
                volume = VALUES(volume);
        """

        total_rows = 0

        # 2. Fetch historical data for every asset
        for asset in assets:

            asset_id = asset["asset_id"]
            ticker = asset["ticker"]

            print(f"\nFetching data for {ticker}...")

            try:
                df = yf.download(
                    ticker,
                    start=START_DATE,
                    auto_adjust=False,
                    multi_level_index=False,
                    progress=False,
                    repair=True
                )

                if df.empty:
                    print(f"No data returned for {ticker}.")
                    continue

                # 3. Flatten columns if a MultiIndex is returned
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                required_columns = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                missing_columns = [
                    col for col in required_columns
                    if col not in df.columns
                ]

                if missing_columns:
                    print(
                        f"Skipping {ticker}. "
                        f"Missing columns: {missing_columns}"
                    )
                    continue

                # 4. Remove rows with missing essential prices
                df = df.dropna(
                    subset=["Open", "High", "Low", "Close"]
                )

                rows = []

                # 5. Convert each trading day into an SQL record
                for trading_date, row in df.iterrows():

                    adjusted_close = None

                    if "Adj Close" in df.columns:
                        adjusted_value = row["Adj Close"]

                        if pd.notna(adjusted_value):
                            adjusted_close = float(adjusted_value)

                    volume = None

                    if pd.notna(row["Volume"]):
                        volume = int(row["Volume"])

                    rows.append((
                        asset_id,
                        trading_date.date(),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        adjusted_close,
                        volume
                    ))

                # 6. Insert the records into MySQL
                if rows:
                    insert_cursor.executemany(insert_query, rows)
                    connection.commit()

                    total_rows += len(rows)

                    print(
                        f"{ticker}: {len(rows)} rows inserted/updated."
                    )

            except Exception as e:
                connection.rollback()
                print(f"Error processing {ticker}: {e}")

        print(f"\nTotal rows processed: {total_rows}")

    except Exception as e:
        connection.rollback()
        print(f"Database error: {e}")

    finally:
        cursor.close()
        connection.close()
        print("MySQL connection closed.")


if __name__ == "__main__":
    fetch_stock_prices()
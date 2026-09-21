
from connection import create_connection


assets = [
    ("AAPL", "Apple Inc.", "Technology", "STOCK"),
    ("MSFT", "Microsoft Corporation", "Technology", "STOCK"),
    ("NVDA", "NVIDIA Corporation", "Technology", "STOCK"),
    ("JPM", "JPMorgan Chase", "Financials", "STOCK"),
    ("JNJ", "Johnson & Johnson", "Healthcare", "STOCK"),
    ("XOM", "Exxon Mobil", "Energy", "STOCK"),
    ("WMT", "Walmart", "Consumer Staples", "STOCK"),
    ("CAT", "Caterpillar Inc.", "Industrials", "STOCK"),
    ("SPY", "SPDR S&P 500 ETF", "Index", "ETF"),
    ("GLD", "SPDR Gold Shares", "Commodities", "ETF")
]


def insert_assets():
    connection = create_connection()

    if connection is None:
        return

    cursor = connection.cursor()

    query = """
        INSERT INTO assets
        (ticker, company_name, sector, asset_type)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            company_name = VALUES(company_name),
            sector = VALUES(sector),
            asset_type = VALUES(asset_type);
    """

    try:
        cursor.executemany(query, assets)
        connection.commit()

        print(f"Inserted or updated {cursor.rowcount} asset records.")

    except Exception as e:
        connection.rollback()
        print(f"Error inserting assets: {e}")

    finally:
        cursor.close()
        connection.close()
        print("Connection closed.")


if __name__ == "__main__":
    insert_assets()
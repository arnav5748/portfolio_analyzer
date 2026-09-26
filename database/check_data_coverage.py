import os
from urllib.parse import urlparse, unquote

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

url = urlparse(os.getenv("DATABASE_URL"))

connection = mysql.connector.connect(
    host=url.hostname,
    port=url.port,
    user=unquote(url.username),
    password=unquote(url.password),
    database=url.path.lstrip("/")
)

cursor = connection.cursor()

# 1. List all assets
print("\n========== ASSETS ==========")

cursor.execute("""
    SELECT asset_id, ticker, company_name, sector
    FROM assets
    ORDER BY asset_id;
""")

for row in cursor.fetchall():
    print(row)

# 2. Stock price coverage
print("\n========== STOCK PRICE COVERAGE ==========")

cursor.execute("""
    SELECT
        MIN(trading_date),
        MAX(trading_date),
        COUNT(DISTINCT asset_id),
        COUNT(*)
    FROM stock_prices;
""")

print("Earliest date | Latest date | Assets | Records")
print(cursor.fetchone())

# 3. Factor returns coverage
print("\n========== FACTOR COVERAGE ==========")

cursor.execute("""
    SELECT
        MIN(factor_date),
        MAX(factor_date),
        COUNT(*)
    FROM factor_returns;
""")

print(cursor.fetchone())

# 4. Macro indicators
print("\n========== MACRO INDICATORS ==========")

cursor.execute("""
    SELECT
        indicator_name,
        MIN(indicator_date),
        MAX(indicator_date),
        COUNT(*)
    FROM macro_indicators
    GROUP BY indicator_name
    ORDER BY indicator_name;
""")

for row in cursor.fetchall():
    print(row)

cursor.close()
connection.close()

print("\n✅ Coverage check completed!")

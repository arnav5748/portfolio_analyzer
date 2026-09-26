import os
from urllib.parse import urlparse, unquote

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL is missing from .env")

url = urlparse(database_url)

host = url.hostname
port = url.port or 3306
user = unquote(url.username) if url.username else None
password = unquote(url.password) if url.password else None
database = url.path.lstrip("/")

print("Host:", host)
print("Port:", port)
print("Database:", database)

connection = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database,
    connection_timeout=15
)

print("\nConnected successfully!\n")

cursor = connection.cursor()

cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]

    print(f"\n--- {table_name} ---")

    cursor.execute(f"DESCRIBE `{table_name}`;")

    for column in cursor.fetchall():
        print(column)

cursor.close()
connection.close()
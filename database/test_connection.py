import os
from urllib.parse import urlparse, unquote

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL is missing from .env")

parsed_url = urlparse(database_url)

try:
    connection = mysql.connector.connect(
        host=parsed_url.hostname,
        port=parsed_url.port,
        user=unquote(parsed_url.username),
        password=unquote(parsed_url.password),
        database=parsed_url.path.lstrip("/"),
        connection_timeout=15
    )

    if connection.is_connected():
        print("Connected successfully to Railway MySQL!")

        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")

        result = cursor.fetchone()
        print("Connected database:", result[0])

        cursor.close()
        connection.close()

except mysql.connector.Error as error:
    print("Database connection failed!")
    print("Error:", error)
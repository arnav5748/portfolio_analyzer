
import mysql.connector
from mysql.connector import Error


def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="SQL_PASSWORD",
            database="portfolio_risk"
        )

        if connection.is_connected():
            print("Successfully connected to MySQL!")
            print("Database: portfolio_risk")

        return connection

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None


if __name__ == "__main__":
    connection = create_connection()

    if connection and connection.is_connected():
        connection.close()
        print("MySQL connection closed.")
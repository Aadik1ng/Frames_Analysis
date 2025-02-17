import psycopg2
from psycopg2 import pool
import time
# Database configuration using the provided credentials
DB_CONFIG = {
    "dbname": "face_rec_1",
    "user": "aaditya",
    "password": "Hxjrhao22abjbkal1Qjz",
    "host": "194.195.112.175",
    "port": "5432"
}
class Database:
    _connection_pool = None

    @staticmethod
    def initialize():
        """Initialize a connection pool to the PostgreSQL database."""
        if Database._connection_pool is None:
            retries = 5
            while retries > 0:
                try:
                    Database._connection_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
                    if Database._connection_pool:
                        print("Database connection pool created successfully.")
                    break
                except Exception as e:
                    print(f"Database connection failed. Retrying... ({retries} left)")
                    time.sleep(3)
                    retries -= 1
                    if retries == 0:
                        raise Exception("Failed to connect to the database after multiple retries") from e

    @staticmethod
    def get_connection():
        """Get a connection from the pool."""
        if Database._connection_pool is None:
            Database.initialize()
        return Database._connection_pool.getconn()

    @staticmethod
    def release_connection(connection):
        """Release a connection back to the pool."""
        Database._connection_pool.putconn(connection)

    @staticmethod
    def close_all_connections():
        """Close all connections in the pool."""
        if Database._connection_pool:
            Database._connection_pool.closeall()

# Initialize the database connection pool at module load.
Database.initialize()

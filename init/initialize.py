from database.db import Database
from data.table_names import create_tables

Database.initialize()
def initialize():
    """Run all one-time initialization functions."""
    print("Initializing database and system setup...")
    create_tables()
    # You can run additional one-time functions here.
    print("Initialization complete.")

if __name__ == "__main__":
    initialize()

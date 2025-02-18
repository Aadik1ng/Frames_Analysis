from database.db import Database
from utils.utils import setup_logger

logger = setup_logger(__name__)

from database.db import Database
from utils.utils import setup_logger

logger = setup_logger(__name__)

def create_embeds_table(embedding_dim=512, table_name="embeds"):
    """
    Create the 'embeds' table in the public schema if it doesn't exist.
    The table will have:
      - id: a serial primary key,
      - embedding: a vector column (requires pgvector),
      - metadata: a JSONB column.
    """
    connection = Database.get_connection()
    try:
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            embedding vector({embedding_dim}),
            metadata JSONB
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        logger.info(f"Table '{table_name}' ensured to exist.")
    except Exception as e:
        logger.error(f"Error creating table '{table_name}': {e}")
        connection.rollback()
    finally:
        Database.release_connection(connection)


def create_tables():
    """
    Create additional necessary SQL tables if they do not already exist.
    This function creates the 'faces' and 'track' tables.
    """
    queries = {
        "faces": """
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                p_id TEXT NOT NULL UNIQUE,
                name TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_faces_pid ON faces(p_id);
        """,
        "track": """
            CREATE TABLE IF NOT EXISTS track (
                id SERIAL PRIMARY KEY,
                req_id TEXT,
                v_id TEXT,
                created_at TIMESTAMP,
                end_time TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_track_req_state ON track(req_id);
        """
    }
    
    conn = None
    try:
        conn = Database.get_connection()
        cursor = conn.cursor()
        for table, query in queries.items():
            logger.info(f"Creating table {table}...")
            cursor.execute(query)
        conn.commit()
        cursor.close()
        logger.info("All additional tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        if conn:
            Database.release_connection(conn)

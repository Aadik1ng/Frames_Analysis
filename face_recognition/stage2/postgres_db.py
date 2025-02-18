import numpy as np
import psycopg2
import json
from utils.utils import setup_logger
from database.db import Database
from data.table_names import create_embeds_table, create_tables

logger = setup_logger(__name__)

# Initialize the database connection pool
Database.initialize()
# Create the necessary tables:
#   - create_embeds_table() creates the 'embeds' table with columns: id, embedding, metadata.
#   - create_tables() creates the 'faces' and 'track' tables.
create_embeds_table()
# create_tables()

def insert_embedding(embedding, meta, embedding_dim=512, table_name="embeds"):
    """
    Insert a single embedding and its metadata into the 'embeds' table.
    
    Parameters:
      - embedding: A list or numpy array of floats.
      - meta: Metadata (should be JSON-serializable).
      - embedding_dim: Expected dimension of the embedding.
      - table_name: The name of the table (default is "embeds").
    """
    # Ensure the embedding is a NumPy array of type float32.
    embedding = np.array(embedding, dtype='float32')
    if embedding.ndim != 1:
        embedding = embedding.flatten()
    if len(embedding) != embedding_dim:
        logger.error(f"Embedding dimension mismatch: Expected {embedding_dim}, got {len(embedding)}")
        return

    vector = embedding.tolist()
    connection = Database.get_connection()
    try:
        cursor = connection.cursor()
        insert_query = f"INSERT INTO {table_name} (embedding, metadata) VALUES (%s, %s);"
        cursor.execute(insert_query, (vector, json.dumps(meta)))
        connection.commit()
        cursor.close()
        logger.info("Inserted embedding successfully.")
    except Exception as e:
        logger.error(f"Error inserting embedding: {e}")
        connection.rollback()
    finally:
        Database.release_connection(connection)

def search_embedding(query_embedding, k=1, embedding_dim=512, table_name="embeds"):
    """
    Search for the k nearest embeddings using the Euclidean (L2) distance provided by pgvector.
    
    Parameters:
      - query_embedding: A list or numpy array of floats (1D).
      - k: Number of nearest neighbors to return.
      - embedding_dim: Expected dimension of the embedding.
      - table_name: The name of the table (default is "embeds").
    
    Returns:
      A list of dictionaries with keys: 'id', 'metadata', and 'distance'.
    """
    query_embedding = np.array(query_embedding, dtype='float32')
    if query_embedding.ndim != 1:
        query_embedding = query_embedding.flatten()
    if len(query_embedding) != embedding_dim:
        logger.error(f"Query embedding dimension mismatch: Expected {embedding_dim}, got {len(query_embedding)}")
        return []

    vector = query_embedding.tolist()
    connection = Database.get_connection()
    results = []
    try:
        cursor = connection.cursor()
        search_query = f"""
        SELECT id, metadata, embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
        """
        cursor.execute(search_query, (vector, vector, k))
        rows = cursor.fetchall()
        for row in rows:
            id_, meta, distance = row
            results.append({"id": id_, "metadata": meta, "distance": distance})
        cursor.close()
        logger.info("Search completed successfully.")
    except Exception as e:
        logger.error(f"Error during search: {e}")
    finally:
        Database.release_connection(connection)
    return results

def get_embeds_count(table_name="embeds"):
    """
    Returns the total number of embeddings stored in the specified table.
    """
    connection = Database.get_connection()
    count = 0
    try:
        cursor = connection.cursor()
        count_query = f"SELECT COUNT(*) FROM {table_name};"
        cursor.execute(count_query)
        count = cursor.fetchone()[0]
        cursor.close()
        logger.info(f"Embeds count retrieved successfully: {count}")
    except Exception as e:
        logger.error(f"Error retrieving embeds count: {e}")
    finally:
        Database.release_connection(connection)
    return count

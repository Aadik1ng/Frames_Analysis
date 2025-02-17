# src/vector_db.py
import numpy as np
import faiss
import pickle
import os
from utils.utils import setup_logger

logger = setup_logger(__name__)

class VectorDB:
    def __init__(self, embedding_dim=512, index_path="vector_db.index", metadata_path="metadata.pkl", index_factory_string="Flat"):
        """
        Initialize the vector database.
        - embedding_dim: dimension of face embeddings.
        - index_path: file path to save/load FAISS index.
        - metadata_path: file path to save/load metadata.
        - index_factory_string: FAISS index type (e.g., "Flat" for exact search).
        """
        self.embedding_dim = embedding_dim  # Use the parameter consistently
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # Exact nearest neighbor search
        self.metadata = []  # Store metadata

        # Load previous embeddings if they exist
        self.load_index()
        logger.info("VectorDB initialized.")

    def add_embedding(self, embedding, meta):
        """
        Add a single embedding with its metadata.
        """
        # Convert to a NumPy array of type float32 and ensure it is at least 2D.
        embedding = np.atleast_2d(np.array(embedding, dtype='float32'))
        
        # Check if the embedding has the correct number of dimensions.
        if embedding.shape[1] != self.embedding_dim:
            logger.error(f"Embedding dimension mismatch: Expected {self.embedding_dim}, got {embedding.shape[1]}. Skipping addition for this keyframe.")
            return  # Skip adding this embedding
        
        # Add the embedding to the FAISS index and store its metadata.
        self.index.add(embedding)
        self.metadata.append(meta)
        logger.info(f"Added embedding with metadata: {meta}")


    def search(self, query_embedding, k=1):
        query_embedding = np.array(query_embedding, dtype='float32')
        query_embedding = np.atleast_2d(query_embedding)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        logger.debug(f"FAISS index dimension (self.index.d): {self.index.d}")
        
        if query_embedding.shape[1] != self.embedding_dim:
            logger.error(
                f"Query embedding dimension mismatch: Expected {self.embedding_dim}, got {query_embedding.shape[1]}. Skipping search for this keyframe."
            )
            return []  # Skip search by returning an empty result

        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], dist))
        logger.debug(f"Search results: {results}")
        return results


    def get_all_embeddings(self):
        """
        Retrieve all stored embeddings.
        """
        if self.index.ntotal == 0:
            return None
        stored_embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, stored_embeddings)
        return stored_embeddings

    def get_metadata(self):
        """
        Retrieve metadata of stored embeddings.
        """
        return self.metadata

    def save_index(self):
        """
        Save the FAISS index and metadata.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("VectorDB saved successfully.")

    def load_index(self):
        """
        Load the FAISS index and metadata if available.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info("FAISS index loaded.")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Metadata loaded.")

# A helper to create or load a vector DB instance.
def create_vector_db(embedding_dim=512):
    return VectorDB(embedding_dim)

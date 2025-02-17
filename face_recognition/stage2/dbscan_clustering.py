# src/dbscan_clustering.py

import numpy as np
from sklearn.cluster import DBSCAN
from utils.utils import setup_logger
logger = setup_logger(__name__)

def cluster_embeddings(embeddings, eps=0.8, min_samples=2):
    logger.info("Starting DBSCAN clustering on embeddings...")
    if isinstance(embeddings, list):
        embeddings = np.vstack(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dbscan.fit(embeddings)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"DBSCAN clustering found {n_clusters} clusters with labels: {labels}")
    return labels, n_clusters

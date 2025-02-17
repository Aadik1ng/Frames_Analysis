# src/face_recognition.py
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from utils.utils import setup_logger

logger = setup_logger(__name__)

# Load the FaceNet model (InceptionResnetV1 pretrained on VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_face(face_img, image_size=160):
    """Preprocess the face image: resize, convert to RGB and normalize."""
    # Resize the face image
    face_resized = cv2.resize(face_img, (image_size, image_size))
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    face_normalized = face_rgb / 255.0
    # Transpose and convert to torch tensor
    face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).float()
    # Normalize using FaceNet's mean and std (if needed, here we assume range [0,1])
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    return face_tensor

def extract_embedding(face_img):
    """Extract face embedding from the face image."""
    face_tensor = preprocess_face(face_img)
    with torch.no_grad():
        embedding = model(face_tensor)
    # Convert to numpy array
    embedding_np = embedding.squeeze(0).cpu().numpy()
    logger.debug("Extracted face embedding.")
    return embedding_np

def compare_embeddings(embedding1, embedding2, threshold=0.6):
    """
    Compute Euclidean distance between embeddings.
    Returns True if embeddings are similar (i.e. same identity) based on threshold.
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    logger.debug(f"Embedding distance: {distance}")
    return distance < threshold, distance

# src/face_detection.py
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from utils.utils import setup_logger

logger = setup_logger(__name__)

# Initialize Haar Cascade classifier for fast detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MTCNN for robust face detection
mtcnn = MTCNN(keep_all=True, device='cpu')  # change device to 'cuda' if available

def detect_faces_haar(frame, scaleFactor=1.1, minNeighbors=5):
    """Detect faces using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    logger.debug(f"Haar detection found {len(faces)} faces")
    return faces  # Returns list of (x, y, w, h)

def detect_faces_mtcnn(frame):
    """Detect faces using MTCNN."""
    # MTCNN expects a PIL image (or numpy array in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb_frame)
    if boxes is None:
        boxes = []
    else:
        boxes = boxes.astype(int)
    logger.debug(f"MTCNN detection found {len(boxes)} faces")
    return boxes, probs

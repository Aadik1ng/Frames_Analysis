# src/facial_landmarks.py
import cv2
import mediapipe as mp
from utils.utils import setup_logger

logger = setup_logger(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(frame):
    """
    Detect facial landmarks using MediaPipe Face Mesh.

    Args:
        frame (numpy.ndarray): BGR image.

    Returns:
        list: Each element is a list of (x, y) tuples representing landmarks for one detected face.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    landmarks_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            # Convert normalized landmark coordinates to pixel coordinates.
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            landmarks_list.append(landmarks)
        logger.debug(f"Detected {len(landmarks_list)} face(s) with landmarks.")
    else:
        logger.debug("No faces detected with MediaPipe Face Mesh.")

    return landmarks_list


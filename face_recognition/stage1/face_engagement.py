import cv2
from .facial_landmarks import extract_landmarks   
from .roi_analysis import is_face_engaged, draw_roi
from .face_detection import detect_faces_mtcnn
from utils.utils import setup_logger

logger = setup_logger(__name__)

def detect_and_filter_engaged_faces(frame):
    """
    Detect faces using MTCNN and check for engagement by verifying that the 
    face’s landmark center lies within the MTCNN bounding box.
    
    Returns:
        engaged_faces (list): A list of tuples (bbox, landmarks) for engaged faces.
    """
    engaged_faces = []
    boxes, _ = detect_faces_mtcnn(frame)
    landmarks_list = extract_landmarks(frame)
    
    for box in boxes:
        # Convert MTCNN box (assumed as [x1, y1, x2, y2]) to (x, y, w, h)
        x1, y1, x2, y2 = box
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Check each set of landmarks to see if its center lies within the bbox.
        for landmarks in landmarks_list:
            xs = [pt[0] for pt in landmarks]
            ys = [pt[1] for pt in landmarks]
            if not xs or not ys:
                continue
            center_x = sum(xs) // len(xs)
            center_y = sum(ys) // len(ys)
            # (Optional) First check that the landmark center is roughly inside the raw box
            if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                if is_face_engaged(landmarks, bbox):
                    engaged_faces.append((bbox, landmarks))
                    logger.info(f"Engaged face detected: BBox {bbox}, Landmark center ({center_x}, {center_y})")
                break
    return engaged_faces

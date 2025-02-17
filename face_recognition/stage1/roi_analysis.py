import cv2
from utils.utils import setup_logger

logger = setup_logger(__name__)

def is_face_engaged(landmarks, bbox):
    """
    Check if the face (represented by landmarks) is engaged by verifying that
    the center of the landmarks is within the provided bounding box (bbox).
    
    Parameters:
        landmarks (list): A list of (x, y) tuples for the facial landmarks.
        bbox (tuple): A tuple (x, y, w, h) where (x, y) is the top-left corner.
        
    Returns:
        bool: True if the face is engaged, False otherwise.
    """
    if not landmarks:
        return False
    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]
    center_x = sum(xs) // len(xs)
    center_y = sum(ys) // len(ys)
    x, y, w, h = bbox
    engaged = (x <= center_x <= x + w) and (y <= center_y <= y + h)
    logger.debug(f"Face center at ({center_x}, {center_y}) is {'inside' if engaged else 'outside'} bbox: {bbox}")
    return engaged

def draw_roi(frame, bbox, color=(255, 0, 0), thickness=2):
    """
    Draw the bounding box (ROI) on the frame.
    
    Parameters:
        frame: The image frame.
        bbox (tuple): Tuple (x, y, w, h) for the ROI.
        color (tuple): The rectangle color.
        thickness (int): Line thickness.
    
    Returns:
        frame: The image frame with the ROI drawn.
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    return frame

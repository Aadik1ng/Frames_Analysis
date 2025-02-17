# face_recognition/stage1/stage1.py

import cv2
from face_recognition.stage1 import face_detection  # adjust the import as needed
from face_recognition.stage1 import facial_landmarks  # adjust the import as needed
from utils.utils import setup_logger

logger = setup_logger(__name__)

def detect_faces_and_landmarks(frame):
    """
    Combines face detection and landmark extraction.
    
    1. Uses MTCNN (or any detection method) to detect face bounding boxes.
    2. Uses MediaPipe to extract facial landmarks.
    
    Returns:
        List of detections, where each detection is a tuple:
        (bounding_box, cropped_face_image, landmarks)
    """
    # Detect faces using MTCNN
    boxes, _ = face_detection.detect_faces_mtcnn(frame)
    
    # Extract landmarks for the entire frame
    landmarks_list = facial_landmarks.extract_landmarks(frame)
    
    detections = []
    
    for box in boxes:
        x, y, w, h = box
        cropped_face = frame[y:y+h, x:x+w]
        
        # Attempt to match this face with one of the landmarks sets.
        # (This is a simple approach: assign the first landmarks set whose center falls inside the box.)
        for landmarks in landmarks_list:
            if not landmarks:
                continue
            xs = [pt[0] for pt in landmarks]
            ys = [pt[1] for pt in landmarks]
            center_x = sum(xs) // len(xs)
            center_y = sum(ys) // len(ys)
            if (x <= center_x <= x+w) and (y <= center_y <= y+h):
                detections.append((box, cropped_face, landmarks))
                logger.info(f"Detected face at {box} with corresponding landmarks.")
                break  # Move on to the next detected face
    
    return detections

# Test the function (if running this module directly)
if __name__ == "__main__":
    import cv2
    test_frame = cv2.imread("sample.jpg")  # Replace with a valid path to a test image
    if test_frame is None:
        raise FileNotFoundError("Sample image not found.")
    dets = detect_faces_and_landmarks(test_frame)
    print("Detections:")
    for det in dets:
        print(det[0])  # Print the bounding box

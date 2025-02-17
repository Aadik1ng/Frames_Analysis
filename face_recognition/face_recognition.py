"""
face_recognition.py

This file contains the high-level face recognition logic.
It integrates Stage 1 (face detection, landmark extraction, engagement filtering)
and Stage 2 (embedding extraction and comparison/clustering).

It serves as the main entry point for the face recognition functionality.
"""

# Import stage 1 functionality (e.g., face detection and landmark extraction)
from face_recognition.stage1.stage1 import detect_faces_and_landmarks

# Import stage 2 functionality (e.g., embedding extraction and comparison)
from .stage2.face_recognition import extract_embedding, compare_embeddings

def recognize_faces(frame, known_face_embeddings):
    """
    Recognize faces in the given frame.

    Stage 1:
        - Detect faces and extract facial landmarks.
        - Return bounding boxes and cropped face images.

    Stage 2:
        - For each detected face, extract an embedding.
        - Compare the embedding against known face embeddings to determine a match.

    :param frame: The input image (numpy array).
    :param known_face_embeddings: A dict mapping person IDs to their face embeddings.
    :return: A list of tuples, each containing:
             (bounding_box, matched_person_id, similarity_score)
    """
    results = []

    # Stage 1: Detect faces and extract landmarks
    # This function should return a list of detections,
    # where each detection is a tuple: (bbox, cropped_face_image, landmarks)
    detections = detect_faces_and_landmarks(frame)

    for detection in detections:
        bbox, face_img, landmarks = detection
        
        # Stage 2: Extract the face embedding for the detected face image
        embedding = extract_embedding(face_img)

        # Compare with known embeddings and find the best match.
        best_match = None
        best_distance = float('inf')
        for person_id, known_embedding in known_face_embeddings.items():
            match, distance = compare_embeddings(embedding, known_embedding)
            # You can further define your threshold logic here.
            if match and distance < best_distance:
                best_distance = distance
                best_match = person_id

        results.append((bbox, best_match, best_distance))

    return results


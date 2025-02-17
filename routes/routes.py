import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from utils.keyframe_extractor import extract_keyframes_skip, extract_keyframes_histogram, extract_keyframes_diff
from face_recognition.stage1.face_detection import detect_faces_mtcnn
from face_recognition.stage1.facial_landmarks import extract_landmarks
from face_recognition.stage1.face_engagement import detect_and_filter_engaged_faces
from face_recognition.stage2.face_recognition import extract_embedding  # using our insightface-based module, for example
from face_recognition.stage2.vector_db import create_vector_db
from utils.utils import setup_logger

logger = setup_logger(__name__)
routes_blueprint = Blueprint("routes", __name__)

@routes_blueprint.route("/process_video", methods=["POST"])
def process_video():
    """
    Processes a video file to extract keyframes, detect faces,
    check for engagement via ROI, extract embeddings, and store unique faces.
    It then generates a screen time analysis report.
    
    Expected multipart/form-data parameters:
      - video: the video file.
      - method: keyframe extraction method ("skip", "histogram", "diff"). Default: "skip"
      - skip_rate: (if method=="skip") number of frames to skip. Default: 50.
      - output_folder: folder to save unique face images. Default: "data/unique_faces"
      - (other parameters can be added as needed)
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)

    # Read extraction parameters
    method = request.form.get("method", "skip").lower()
    output_folder = request.form.get("output_folder", "data/unique_faces")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        if method == "skip":
            skip_rate = int(request.form.get("skip_rate", 50))
            keyframes = extract_keyframes_skip(temp_video_path, skip_rate=skip_rate)
        elif method == "histogram":
            threshold = float(request.form.get("threshold", 0.5))
            keyframes = extract_keyframes_histogram(temp_video_path, threshold=threshold)
        elif method == "diff":
            pixel_threshold = int(request.form.get("pixel_threshold", 30))
            diff_threshold = int(request.form.get("diff_threshold", 50000))
            keyframes = extract_keyframes_diff(temp_video_path, pixel_threshold, diff_threshold)
        else:
            return jsonify({"error": "Unknown extraction method"}), 400
    except Exception as e:
        os.remove(temp_video_path)
        return jsonify({"error": f"Keyframe extraction failed: {str(e)}"}), 500

    # Initialize (or load) the FAISS vector DB for embeddings
    vector_db = create_vector_db(512)  # assumes 512-dim embeddings

    # Dictionary to track how many keyframes each face appears in.
    face_appearance = {}  # face_id -> count
    unique_faces_results = []  # list of dicts with face metadata

    # Process each keyframe
    for frame_index, frame in keyframes:
        # Use face detection (MTCNN) and landmark extraction to determine engaged faces.
        # Here we use our helper from face_engagement that internally uses both.
        engaged_faces = detect_and_filter_engaged_faces(frame)
        
        for box, landmarks in engaged_faces:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
            # Extract embedding for the face
            embedding = extract_embedding(face_img)
            if embedding is None:
                continue

            # Check for uniqueness: search in the vector DB for similar embeddings.
            search_result = vector_db.search(embedding, k=1)
            is_unique = True
            if search_result:
                # If a similar embedding exists, check the distance.
                meta, dist = search_result[0]
                if dist < 0.6:  # threshold can be parameterized
                    is_unique = False
                    face_id = meta.get("face_id", "unknown")
            if is_unique:
                # Assign a unique face ID
                face_id = f"face_{len(vector_db.metadata) + 1}"
                meta = {"face_id": face_id, "frame_index": frame_index}
                vector_db.add_embedding(embedding, meta)
                
                # Save the unique face image
                out_path = os.path.join(output_folder, f"{face_id}.jpg")
                cv2.imwrite(out_path, face_img)
                unique_faces_results.append({
                    "face_id": face_id,
                    "frame_index": frame_index,
                    "saved_path": out_path
                })
            
            # Update appearance count (whether new or duplicate)
            face_appearance[face_id] = face_appearance.get(face_id, 0) + 1

    # Persist the vector DB (this could also be integrated with PostgreSQL)
    vector_db.save_index()

    # Generate a screen time analysis report based on keyframe counts
    total_keyframes = len(keyframes)
    report_lines = ["===== Screen Time Analysis Report =====",
                    f"Total keyframes processed: {total_keyframes}"]
    for face_id, count in face_appearance.items():
        report_lines.append(f"{face_id}: Appeared in {count} keyframes")
    report = "\n".join(report_lines)
    logger.info(report)

    # Clean up temporary video file
    os.remove(temp_video_path)

    return jsonify({
        "unique_faces": unique_faces_results,
        "screen_time_report": report
    })

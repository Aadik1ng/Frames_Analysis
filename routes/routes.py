import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from utils.keyframe_extractor import (
    extract_keyframes_skip,
    extract_keyframes_histogram,
    extract_keyframes_diff
)
from face_recognition.stage1.face_detection import detect_faces_mtcnn
from face_recognition.stage1.facial_landmarks import extract_landmarks
from face_recognition.stage1.face_engagement import detect_and_filter_engaged_faces
from face_recognition.stage2.face_recognition import extract_embedding  # e.g., using our insightface-based module
from face_recognition.stage2.postgres_db import (
    create_embeds_table,
    insert_embedding,
    search_embedding,
    get_embeds_count
)
from utils.utils import setup_logger

logger = setup_logger(__name__)
routes_blueprint = Blueprint("routes", __name__)

@routes_blueprint.route("/process_video", methods=["POST"])
def process_video():
    """
    Processes an uploaded video (directly from memory) to extract keyframes, detect faces,
    check for engagement, extract embeddings, and store unique faces in PostgreSQL (public.embeds).
    It then generates a screen time analysis report.

    Expected multipart/form-data parameters:
      - video: the video file.
      - method: keyframe extraction method ("skip", "histogram", "diff"). Default: "skip"
      - skip_rate: (if method=="skip") number of frames to skip. Default: 50.
      - output_folder: folder to save unique face images. Default: "data/unique_faces"
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Get the video file from the request (processing directly from memory)
    video_file = request.files['video']

    # Read extraction parameters
    method = request.form.get("method", "skip").lower()
    output_folder = request.form.get("output_folder", "data/unique_faces")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # NOTE: The keyframe extraction functions below must support file-like objects.
        if method == "skip":
            skip_rate = int(request.form.get("skip_rate", 50))
            keyframes = extract_keyframes_skip(video_file, skip_rate=skip_rate)
        elif method == "histogram":
            threshold = float(request.form.get("threshold", 0.5))
            keyframes = extract_keyframes_histogram(video_file, threshold=threshold)
        elif method == "diff":
            pixel_threshold = int(request.form.get("pixel_threshold", 30))
            diff_threshold = int(request.form.get("diff_threshold", 50000))
            keyframes = extract_keyframes_diff(video_file, pixel_threshold, diff_threshold)
        else:
            return jsonify({"error": "Unknown extraction method"}), 400
    except Exception as e:
        return jsonify({"error": f"Keyframe extraction failed: {str(e)}"}), 500

    # Set embedding dimension and table name (using the 'public.embeds' table)
    embedding_dim = 512  # Assuming 512-dim embeddings
    table_name = "embeds"
    create_embeds_table(embedding_dim=embedding_dim, table_name=table_name)

    # Dictionary to track how many keyframes each face appears in.
    face_appearance = {}  # face_id -> count
    unique_faces_results = []  # List of dicts with face metadata

    # Process each keyframe
    for frame_index, frame in keyframes:
        # Detect engaged faces (using your ROI-based engagement detection)
        engaged_faces = detect_and_filter_engaged_faces(frame)
        
        for box, landmarks in engaged_faces:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
            # Extract embedding for the face image.
            embedding = extract_embedding(face_img)
            if embedding is None:
                continue

            # Search PostgreSQL for a similar embedding.
            search_result = search_embedding(
                embedding, k=1, embedding_dim=embedding_dim, table_name=table_name
            )
            is_unique = True
            face_id = None
            if search_result:
                # If a similar embedding exists, check the distance.
                result = search_result[0]
                meta = result.get("metadata", {})
                dist = result.get("distance", float("inf"))
                if dist < 0.6:  # Threshold can be adjusted as needed.
                    is_unique = False
                    face_id = meta.get("face_id", "unknown")
            
            if is_unique:
                # Determine a new unique face ID based on current count.
                current_count = get_embeds_count(table_name=table_name)
                face_id = f"face_{current_count + 1}"
                meta = {"face_id": face_id, "frame_index": frame_index}
                insert_embedding(embedding, meta, embedding_dim=embedding_dim, table_name=table_name)
                
                # Save the unique face image.
                out_path = os.path.join(output_folder, f"{face_id}.jpg")
                cv2.imwrite(out_path, face_img)
                unique_faces_results.append({
                    "face_id": face_id,
                    "frame_index": frame_index,
                    "saved_path": out_path
                })
            
            # Update appearance count (whether new or duplicate).
            if face_id:
                face_appearance[face_id] = face_appearance.get(face_id, 0) + 1

    # Generate a screen time analysis report based on keyframe counts.
    total_keyframes = len(keyframes)
    report_lines = [
        "===== Screen Time Analysis Report =====",
        f"Total keyframes processed: {total_keyframes}"
    ]
    for face_id, count in face_appearance.items():
        report_lines.append(f"{face_id}: Appeared in {count} keyframes")
    report = "\n".join(report_lines)
    logger.info(report)

    return jsonify({
        "unique_faces": unique_faces_results,
        "screen_time_report": report
    })

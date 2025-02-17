# src/screen_time_analysis.py

import cv2
import time
import numpy as np
from utils.video_loader import load_video, get_video_properties
from face_recognition.stage1.face_detection import detect_faces_mtcnn
from face_recognition.stage2.face_recognition import extract_embedding, compare_embeddings
from utils.utils import setup_logger

#from src.face_recognition import extract_embedding, compare_embeddings
from src.utils import setup_logger

logger = setup_logger(__name__, level=20)  # INFO level

# === Configuration Parameters ===
# Threshold for matching embeddings (tune as needed)
EMBEDDING_MATCH_THRESHOLD = 0.6    
# Consider a face appearance as a new sequence if it reappears after this many frames.
SEQUENCE_GAP_THRESHOLD = 2         

class FaceMetricsAggregator:
    """
    Aggregates metrics for each detected face over the duration of the video.
    Metrics include:
      - Screen time (number of frames detected, later converted to seconds)
      - Average detection confidence
      - Average horizontal position (normalized 0=left, 1=right)
      - Frequency (number of distinct appearance sequences)
    """
    def __init__(self, fps):
        """
        Args:
            fps (float): Frames per second of the video.
        """
        self.fps = fps
        self.face_data = {}  # key: face_id, value: metrics dictionary
        self.next_face_id = 1

    def _match_face(self, embedding):
        """
        Compare the provided embedding with stored face embeddings.
        Returns:
            face_id (int) if a match is found; otherwise, returns None.
        """
        for face_id, data in self.face_data.items():
            stored_embedding = data["embedding"]
            is_match, distance = compare_embeddings(embedding, stored_embedding, threshold=EMBEDDING_MATCH_THRESHOLD)
            if is_match:
                return face_id
        return None

    def update_face(self, embedding, conf, center_x_norm, current_frame):
        """
        Updates (or creates) the record for a detected face.
        
        Args:
            embedding (np.array): Face embedding vector.
            conf (float): Detection confidence score.
            center_x_norm (float): Normalized horizontal center (0=left, 1=right).
            current_frame (int): Index of the current video frame.
        """
        face_id = self._match_face(embedding)
        if face_id is None:
            # No match found; create a new face record.
            face_id = self.next_face_id
            self.next_face_id += 1
            self.face_data[face_id] = {
                "embedding": embedding,
                "frame_count": 0,
                "conf_sum": 0.0,
                "conf_count": 0,
                "pos_sum": 0.0,
                "pos_count": 0,
                "sequences": 0,
                "last_seen": None,  # Last frame number when detected.
            }
            logger.info(f"New face {face_id} registered.")

        data = self.face_data[face_id]
        # Update metrics
        data["frame_count"] += 1
        data["conf_sum"] += conf
        data["conf_count"] += 1
        data["pos_sum"] += center_x_norm
        data["pos_count"] += 1

        # Increase sequence count if this is the start of a new appearance
        if data["last_seen"] is None or (current_frame - data["last_seen"]) > SEQUENCE_GAP_THRESHOLD:
            data["sequences"] += 1

        data["last_seen"] = current_frame
        # Optionally update the stored embedding (here we use the latest)
        data["embedding"] = embedding

    def get_aggregated_metrics(self):
        """
        Finalize and return the aggregated metrics for all faces.
        
        Returns:
            dict: Keys are face IDs and values are metrics dictionaries.
        """
        results = {}
        for face_id, data in self.face_data.items():
            screen_time_sec = data["frame_count"] / self.fps
            avg_conf = data["conf_sum"] / data["conf_count"] if data["conf_count"] > 0 else 0.0
            avg_position = data["pos_sum"] / data["pos_count"] if data["pos_count"] > 0 else 0.0
            results[face_id] = {
                "screen_time_sec": screen_time_sec,
                "avg_confidence": avg_conf,
                "avg_horizontal_position": avg_position,
                "sequences": data["sequences"],
                "frame_count": data["frame_count"],
            }
        return results

class ScreenTimeAnalyzer:
    """
    Processes a video file to compute screen time and additional metrics for each face.
    """
    def __init__(self, video_path):
        self.video_path = video_path

    def analyze(self):
        start_time = time.time()
        cap = load_video(self.video_path)
        fps, total_frames = get_video_properties(cap)
        logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

        aggregator = FaceMetricsAggregator(fps)
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            # Use robust face detection (MTCNN)
            boxes, probs = detect_faces_mtcnn(frame)
            if boxes is None or len(boxes) == 0:
                continue

            h, w = frame.shape[:2]
            for i, box in enumerate(boxes):
                x, y, bw, bh = box
                # Get detection confidence (default to 0 if not available)
                conf = probs[i] if probs is not None else 0.0

                # Extract face region and skip invalid ones
                face_img = frame[y:y+bh, x:x+bw]
                if face_img.size == 0:
                    continue

                # Compute face embedding
                embedding = extract_embedding(face_img)
                # Compute normalized horizontal center position (0=left, 1=right)
                center_x = x + bw / 2.0
                center_x_norm = center_x / w

                aggregator.update_face(embedding, conf, center_x_norm, frame_index)

            if frame_index % 50 == 0:
                logger.info(f"Processed frame {frame_index}/{total_frames}")

        cap.release()
        processing_time = time.time() - start_time

        # Aggregate results and rank faces by screen time (highest first)
        results = aggregator.get_aggregated_metrics()
        ranked = sorted(results.items(), key=lambda x: x[1]["screen_time_sec"], reverse=True)

        # Build a formatted report
        report_lines = []
        report_lines.append("===== Screen Time Analysis Report =====")
        report_lines.append(f"Video: {self.video_path}")
        report_lines.append(f"Total frames processed: {frame_index}")
        report_lines.append(f"Video FPS: {fps}")
        report_lines.append(f"Overall processing time: {processing_time:.2f} seconds")
        report_lines.append("")
        report_lines.append("Face Metrics (Ranked by Screen Time):")
        for face_id, metrics in ranked:
            report_lines.append(
                f"Face {face_id}: Screen Time = {metrics['screen_time_sec']:.2f} sec, "
                f"Avg Confidence = {metrics['avg_confidence']:.2f}, "
                f"Avg Horizontal Pos = {metrics['avg_horizontal_position']:.2f}, "
                f"Sequences = {metrics['sequences']}, "
                f"Frame Count = {metrics['frame_count']}"
            )
        report = "\n".join(report_lines)
        logger.info(report)
        return results, report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screen Time Analysis & Data Aggregation")
    parser.add_argument("video_path", help="Path to the input video file")
    args = parser.parse_args()

    analyzer = ScreenTimeAnalyzer(args.video_path)
    results, report = analyzer.analyze()
    print(report)

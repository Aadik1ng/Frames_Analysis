# src/face_tracker.py
import cv2
from utils.video_loader import load_video, get_video_properties
from .face_engagement import detect_and_filter_engaged_faces
from .roi_analysis import draw_roi
from utils.utils import setup_logger

logger = setup_logger(__name__)

class FaceTracker:
    def __init__(self, detection_interval=30):
        self.detection_interval = detection_interval
        self.tracks = {}  # Dictionary to store tracked face data

    def update(self, prev_frame, frame):
        """
        Simulated tracking update function. This should integrate DeepSORT or Optical Flow.
        """
        # TODO: Implement DeepSORT-based face tracking.
        # For now, returning an empty dictionary
        return self.tracks

    def run(self, video_path):
        cap = load_video(video_path)
        fps, total_frames = get_video_properties(cap)
        
        ret, prev_frame = cap.read()
        if not ret:
            logger.error("Could not read the first frame of the video.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update the tracker
            tracks = self.update(prev_frame, frame)
            
            # Draw tracked faces
            for track_id, track in tracks.items():
                x, y, w, h = track['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detect engaged faces
            engaged_faces = detect_and_filter_engaged_faces(frame)
            for box, landmarks in engaged_faces:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw ROI
            frame_with_roi = draw_roi(frame.copy())
            
            cv2.imshow("Face Tracking & Engagement", frame_with_roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_frame = frame.copy()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"D:\person_track\input_video.mp4"
    tracker = FaceTracker(detection_interval=30)
    tracker.run(video_path)

# src/video_loader.py
import cv2
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise IOError(f"Cannot open video: {video_path}")
    logger.info(f"Video {video_path} loaded successfully.")
    return cap

def get_video_properties(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"FPS: {fps}, Total frames: {total_frames}")
    return fps, total_frames

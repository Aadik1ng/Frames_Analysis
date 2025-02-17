# src/keyframe_extractor.py
import cv2
import numpy as np
import os
from utils.utils import setup_logger
logger = setup_logger(__name__)
folder_path = "data/keyframes_folder"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
def compute_histogram(frame, hist_size=256):
    # Convert to HSV for better color distribution analysis if needed
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculate the histogram for the Hue channel
    hist = cv2.calcHist([hsv], [0], None, [hist_size], [0, 180])
    cv2.normalize(hist, hist)
    return hist

def extract_keyframes_histogram(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("No frames read from the video.")
        return keyframes

    prev_hist = compute_histogram(prev_frame)
    frame_index = 1  # starting index
    keyframes.append((0, prev_frame))  # consider the first frame a keyframe

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_hist = compute_histogram(frame)
        # Compute correlation between histograms
        score = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        logger.debug(f"Frame {frame_index}: histogram correlation = {score}")
        # If the correlation is below a threshold, mark as a keyframe
        if score < threshold:
            keyframes.append((frame_index, frame))
            logger.info(f"Keyframe found at frame {frame_index} with score {score}")
        prev_hist = curr_hist
        frame_index += 1
    cap.release()
    return keyframes
def extract_keyframes_diff(video_path, pixel_threshold=30, diff_threshold=50000):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("No frames read from the video.")
        return keyframes

    frame_index = 1
    keyframes.append((0, prev_frame))  # starting frame as keyframe

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Compute absolute difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        # Convert to grayscale and threshold the diff image
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh_diff = cv2.threshold(gray_diff, pixel_threshold, 255, cv2.THRESH_BINARY)
        # Sum the differences
        diff_sum = np.sum(thresh_diff)
        logger.debug(f"Frame {frame_index}: diff sum = {diff_sum}")

        if diff_sum > diff_threshold:
            keyframes.append((frame_index, curr_frame))
            logger.info(f"Keyframe found at frame {frame_index} with diff sum {diff_sum}")
        prev_frame = curr_frame
        frame_index += 1

    cap.release()
    return keyframes
def extract_keyframes_skip(video_path, skip_rate=1000):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_rate == 0:
            keyframes.append((frame_index, frame))
            logger.info(f"Frame {frame_index} captured via skipping.")
        frame_index += 1

    cap.release()
    return keyframes
def extract_keyframes(video_path, method='histogram', **kwargs):
    if method == 'histogram':
        return extract_keyframes_histogram(video_path, **kwargs)
    elif method == 'diff':
        return extract_keyframes_diff(video_path, **kwargs)
    elif method == 'skip':
        return extract_keyframes_skip(video_path, **kwargs)
    else:
        logger.error(f"Unknown extraction method: {method}")
        raise ValueError(f"Unknown extraction method: {method}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Keyframe Extraction")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--method", choices=["histogram", "diff", "skip"], default="histogram", help="Keyframe extraction method")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for histogram method")
    parser.add_argument("--pixel_threshold", type=int, default=30, help="Pixel threshold for diff method")
    parser.add_argument("--diff_threshold", type=int, default=50000, help="Difference sum threshold for diff method")
    parser.add_argument("--skip_rate", type=int, default=10, help="Frame skip rate for skip method")
    parser.add_argument("--output_folder", default="data/keyframes_folder", help="Folder to save keyframe images")

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f"Created output folder: {args.output_folder}")

    # Extract keyframes using the chosen method
    if args.method == "histogram":
        keyframes = extract_keyframes(args.video_path, method="histogram", threshold=args.threshold)
    elif args.method == "diff":
        keyframes = extract_keyframes(args.video_path, method="diff", pixel_threshold=args.pixel_threshold, diff_threshold=args.diff_threshold)
    elif args.method == "skip":
        keyframes = extract_keyframes(args.video_path, method="skip", skip_rate=args.skip_rate)
    else:
        logger.error("Unsupported method selected.")
        raise ValueError("Unsupported method selected.")

    logger.info(f"Extracted {len(keyframes)} keyframes.")

    # Save the keyframes into the specified folder
    for frame_no, frame in keyframes:
        filename = os.path.join(args.output_folder, f"keyframe_{frame_no}.jpg")
        cv2.imwrite(filename, frame)
        logger.info(f"Saved keyframe {frame_no} to {filename}")
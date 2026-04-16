## surya - Video keyframe extraction using hybrid approach
## Scene-change detection + minimum interval sampling + CLIP deduplication
import os
import cv2
import numpy as np
from pathlib import Path


def compute_hist_diff(frame1, frame2):
    """Compute histogram difference between two frames (0.0 = identical, 1.0 = completely different)."""
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 1.0 - score  # 0 = same, 1 = different


def extract_keyframes(video_path, scene_threshold=0.4, min_interval_sec=3.0,
                      max_frames=50):
    """
    Extract keyframes from a video using hybrid approach:
    1. Scene-change detection (histogram diff > threshold)
    2. Minimum interval sampling (fallback every min_interval_sec)
    3. Capped at max_frames total

    Returns:
        list of dicts: [{"frame": np.ndarray (BGR), "timestamp_sec": float, "frame_index": int}]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    min_interval_frames = int(min_interval_sec * fps) if fps > 0 else 90

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO]   FPS={fps:.1f}, frames={total_frames}, duration={duration_sec:.1f}s")

    keyframes = []
    prev_frame = None
    last_keyframe_idx = -min_interval_frames  # ensure first frame is captured

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps if fps > 0 else 0
        is_keyframe = False

        if prev_frame is None:
            # Always capture first frame
            is_keyframe = True
        else:
            frames_since_last = frame_idx - last_keyframe_idx

            # Scene change detection
            diff = compute_hist_diff(prev_frame, frame)
            if diff > scene_threshold and frames_since_last > int(fps * 0.5):
                is_keyframe = True

            # Minimum interval fallback
            elif frames_since_last >= min_interval_frames:
                is_keyframe = True

        if is_keyframe:
            keyframes.append({
                "frame": frame.copy(),
                "timestamp_sec": round(timestamp_sec, 2),
                "frame_index": frame_idx
            })
            last_keyframe_idx = frame_idx

            if len(keyframes) >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames}, stopping extraction")
                break

        prev_frame = frame
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {len(keyframes)} keyframes from {video_path}")
    return keyframes


def save_keyframes(keyframes, output_dir, video_name):
    """Save extracted keyframes as images to disk.

    Returns:
        list of dicts: [{"image_path": str, "timestamp_sec": float, "frame_index": int, "video_name": str}]
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i, kf in enumerate(keyframes):
        filename = f"{video_name}_frame_{i:04d}_{kf['timestamp_sec']:.1f}s.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, kf["frame"])
        saved.append({
            "image_path": filepath,
            "timestamp_sec": kf["timestamp_sec"],
            "frame_index": kf["frame_index"],
            "video_name": video_name
        })
    print(f"[INFO] Saved {len(saved)} keyframes to {output_dir}")
    return saved


def load_videos(data_dir):
    """Find all video files in the data directory."""
    data_path = Path(data_dir).resolve()
    videos = []
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    for ext in video_extensions:
        for vid_file in data_path.glob(f"**/{ext}"):
            print(f"[DEBUG] Found Video: {vid_file}")
            videos.append(str(vid_file))
    return videos


def extract_all_videos(data_dir, output_dir="video_frames", scene_threshold=0.4,
                       min_interval_sec=3.0, max_frames_per_video=50):
    """Extract keyframes from all videos in data_dir.

    Returns:
        list of dicts with metadata for each saved frame.
    """
    video_paths = load_videos(data_dir)
    if not video_paths:
        print(f"[WARN] No videos found in '{data_dir}'")
        return []

    all_frames = []
    for vpath in video_paths:
        video_name = Path(vpath).stem
        keyframes = extract_keyframes(
            vpath,
            scene_threshold=scene_threshold,
            min_interval_sec=min_interval_sec,
            max_frames=max_frames_per_video
        )
        if keyframes:
            saved = save_keyframes(keyframes, output_dir, video_name)
            # Add source video path to each frame's metadata
            for s in saved:
                s["video_path"] = vpath
            all_frames.extend(saved)

    print(f"[INFO] Total keyframes extracted from all videos: {len(all_frames)}")
    return all_frames
## surya end

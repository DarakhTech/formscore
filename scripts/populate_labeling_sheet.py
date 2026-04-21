# scripts/populate_labeling_sheet.py
"""
Runs rep_segmenter on all real squat videos across all sources
and generates a CSV pre-populated with columns A-D ready for labeling.

Output: results/labeling_sheet.csv
Open in Google Sheets: File → Import → Upload
"""

import os
import sys
import csv
import glob
import numpy as np
import cv2
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarkerOptions, PoseLandmarker, RunningMode
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.normalizer import normalize

# ── Config ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker.task")
OUT_CSV    = "results/labeling_sheet.csv"

VIDEO_SOURCES = {
    "workoutfitness": "data/workoutfitness-video/squat/*.mp4",
    "similar_dataset": "data/real-time-exercise-recognition-dataset/similar_dataset/squat/*.mp4",
    "final_kaggle":   "data/real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat/*.mp4",
    "my_test":        "data/real-time-exercise-recognition-dataset/my_test_video_1/squat/*.mp4",
}

COLUMNS = [
    "video_source", "video_filename", "rep_number", "rep_start_frame",
    "rep_end_frame", "rep_duration_s", "rater",
    "depth", "knee_tracking", "spine", "hip_symmetry", "knee_symmetry",
    "descent_control", "ascent_drive", "tempo", "foot_position", "head_neck",
    "total_score", "notes", "exclude"
]


# ── Model download ────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        print("  Downloading MediaPipe model...")
        urllib.request.urlretrieve(url, MODEL_PATH)


# ── Pose extraction ───────────────────────────────────────
def extract_landmarks(video_path: str) -> np.ndarray:
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_lm     = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            lm_array = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                dtype=np.float32
            )
        else:
            lm_array = np.zeros((33, 4), dtype=np.float32)
        all_lm.append(lm_array)

    cap.release()
    landmarker.close()
    return np.stack(all_lm) if all_lm else np.zeros((0, 33, 4)), fps


# ── Rep segmentation ──────────────────────────────────────
def segment_reps(landmarks_raw: np.ndarray,
                 fps: float,
                 smooth_sigma: float = 3.0,
                 min_distance: int = 30) -> list[tuple[int, int]]:
    if len(landmarks_raw) == 0:
        return []

    hip_mid_y    = (landmarks_raw[:, 23, 1] + landmarks_raw[:, 24, 1]) / 2.0
    smoothed     = gaussian_filter1d(hip_mid_y, sigma=smooth_sigma)
    signal_range = smoothed.max() - smoothed.min()

    if signal_range < 0.01:
        return []

    prominence = signal_range * 0.30

    bottoms, _ = find_peaks( smoothed, distance=min_distance, prominence=prominence)
    tops,    _ = find_peaks(-smoothed, distance=min_distance, prominence=prominence)

    if len(bottoms) == 0 or len(tops) == 0:
        return []

    reps = []
    for bottom in bottoms:
        before = tops[tops < bottom]
        after  = tops[tops > bottom]
        start  = int(before[-1]) if len(before) > 0 else 0
        end    = int(after[0])   if len(after)  > 0 else len(hip_mid_y) - 1
        if end > start + 10:
            reps.append((start, end))

    reps   = sorted(set(reps))
    if not reps:
        return []
    merged = [reps[0]]
    for r in reps[1:]:
        if r[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)

    return merged


# ── Main ──────────────────────────────────────────────────
def main():
    download_model()
    os.makedirs("results", exist_ok=True)

    rows      = []
    total_vids = 0
    total_reps = 0

    for source, pattern in VIDEO_SOURCES.items():
        videos = sorted(glob.glob(pattern))
        if not videos:
            print(f"  {source}: no videos found")
            continue

        print(f"\n=== {source} ({len(videos)} videos) ===")

        for vpath in videos:
            fname = os.path.basename(vpath)
            print(f"  {fname} ... ", end="", flush=True)

            try:
                landmarks, fps = extract_landmarks(vpath)

                if len(landmarks) == 0:
                    print("no frames — skipped")
                    continue

                reps = segment_reps(landmarks, fps)
                print(f"{len(reps)} reps detected")

                if len(reps) == 0:
                    # Still add one row so labeler can review the video
                    rows.append({
                        "video_source":    source,
                        "video_filename":  fname,
                        "rep_number":      0,
                        "rep_start_frame": 0,
                        "rep_end_frame":   len(landmarks) - 1,
                        "rep_duration_s":  round(len(landmarks) / fps, 1),
                        "rater":           "",
                        "depth": "", "knee_tracking": "", "spine": "",
                        "hip_symmetry": "", "knee_symmetry": "",
                        "descent_control": "", "ascent_drive": "",
                        "tempo": "", "foot_position": "", "head_neck": "",
                        "total_score":     "=SUM(H{r}:Q{r})",
                        "notes":           "no reps detected — review manually",
                        "exclude":         "TRUE",
                    })
                    continue

                for i, (start, end) in enumerate(reps, 1):
                    dur = round((end - start) / fps, 1)
                    rows.append({
                        "video_source":    source,
                        "video_filename":  fname,
                        "rep_number":      i,
                        "rep_start_frame": start,
                        "rep_end_frame":   end,
                        "rep_duration_s":  dur,
                        "rater":           "",
                        "depth": "", "knee_tracking": "", "spine": "",
                        "hip_symmetry": "", "knee_symmetry": "",
                        "descent_control": "", "ascent_drive": "",
                        "tempo": "", "foot_position": "", "head_neck": "",
                        "total_score":     "",
                        "notes":           "",
                        "exclude":         "",
                    })
                    total_reps += 1

                total_vids += 1

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Done ===")
    print(f"  Videos processed : {total_vids}")
    print(f"  Reps detected    : {total_reps}")
    print(f"  Total rows       : {len(rows)}")
    print(f"  Saved            : {OUT_CSV}")
    print(f"\n  Import into Google Sheets:")
    print(f"  File → Import → Upload → {OUT_CSV}")
    print(f"  Then share with Harsha and Mihir")


if __name__ == "__main__":
    main()
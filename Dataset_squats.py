from pathlib import Path
from typing import Optional, Tuple, List, Dict
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "pose_landmarker.task"

SQUAT_FOLDERS = [
    "workoutfitness-video/squat",
    "real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat",
]

METADATA_CSV = "squat_video_metadata.csv"

TASK1_OUTPUT_DIR = Path("processed_real_squat")
TASK2_OUTPUT_DIR = Path("segmented_squat_reps")

TASK1_OUTPUT_DIR.mkdir(exist_ok=True)
TASK2_OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_FRAMES = 60
VIDEO_EXTENSIONS = {".mp4", ".mov"}

# Shared processing
MIN_VALID_RATIO = 0.50

# Task 2 tuned settings
MIN_REP_FRAMES = 22
MAX_REP_FRAMES = 180
MIN_BOTTOM_DISTANCE = 16
TOP_SEARCH_PAD = 12


# ============================================================
# BLAZEPOSE LANDMARK INDEXES
# ============================================================
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ============================================================
# GENERAL HELPERS
# ============================================================
def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(v) + eps)


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = safe_norm(ba) * safe_norm(bc)
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def resample_sequence(seq: np.ndarray, target_len: int = 60) -> np.ndarray:
    if len(seq) == 0:
        raise ValueError("Cannot resample empty sequence.")
    if len(seq) == target_len:
        return seq.astype(np.float32)

    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(x_new, x_old, seq[:, d])
    return out


def smooth_signal(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    if n < 7:
        return signal.copy()

    window = min(21, n if n % 2 == 1 else n - 1)
    if window < 5:
        return signal.copy()

    return savgol_filter(signal, window_length=window, polyorder=2)


def find_local_extreme(signal: np.ndarray, center: int, pad: int, mode: str = "max") -> int:
    left = max(0, center - pad)
    right = min(len(signal), center + pad + 1)
    window = signal[left:right]

    if len(window) == 0:
        return center

    if mode == "max":
        return left + int(np.argmax(window))
    elif mode == "min":
        return left + int(np.argmin(window))
    else:
        raise ValueError("mode must be 'max' or 'min'")


# ============================================================
# TASK 0: METADATA / LOADER
# ============================================================
def build_squat_metadata(
    squat_folders: List[str],
    output_csv: str = METADATA_CSV
) -> pd.DataFrame:
    records = []

    for folder_str in squat_folders:
        folder = Path(folder_str)

        if not folder.exists():
            print(f"Warning: folder not found -> {folder}")
            continue

        dataset_source = folder.parent.name

        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append({
                    "video_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "exercise_label": "squat",
                    "dataset_source": dataset_source,
                    "extension": file_path.suffix.lower(),
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No squat videos found. Check SQUAT_FOLDERS paths.")

    df = df.sort_values(by=["dataset_source", "file_name"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved metadata to {output_csv}")
    print(f"Total squat videos found: {len(df)}")
    print("\nCount by dataset source:")
    print(df["dataset_source"].value_counts())

    return df


# ============================================================
# BLAZEPOSE / FEATURE EXTRACTION
# ============================================================
def get_pose_landmarker(model_path: str) -> vision.PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks_from_frame(
    frame_bgr: np.ndarray,
    landmarker: vision.PoseLandmarker
) -> Optional[np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    pose = result.pose_landmarks[0]
    arr = np.zeros((33, 4), dtype=np.float32)

    for i, lm in enumerate(pose):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
        arr[i, 3] = getattr(lm, "visibility", 1.0)

    return arr


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    xyv = landmarks.copy()

    l_hip = xyv[LEFT_HIP, :3]
    r_hip = xyv[RIGHT_HIP, :3]
    l_sh = xyv[LEFT_SHOULDER, :3]
    r_sh = xyv[RIGHT_SHOULDER, :3]

    hip_mid = (l_hip + r_hip) / 2.0
    shoulder_mid = (l_sh + r_sh) / 2.0

    torso_scale = max(safe_norm(shoulder_mid - hip_mid), 1e-6)
    xyv[:, :3] = (xyv[:, :3] - hip_mid) / torso_scale
    return xyv


def compute_8_features(norm_lm: np.ndarray) -> np.ndarray:
    ls = norm_lm[LEFT_SHOULDER, :3]
    rs = norm_lm[RIGHT_SHOULDER, :3]
    lh = norm_lm[LEFT_HIP, :3]
    rh = norm_lm[RIGHT_HIP, :3]
    lk = norm_lm[LEFT_KNEE, :3]
    rk = norm_lm[RIGHT_KNEE, :3]
    la = norm_lm[LEFT_ANKLE, :3]
    ra = norm_lm[RIGHT_ANKLE, :3]
    lheel = norm_lm[LEFT_HEEL, :3]
    rheel = norm_lm[RIGHT_HEEL, :3]

    hip_mid = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0

    left_knee_angle = angle_3pt(lh, lk, la)
    right_knee_angle = angle_3pt(rh, rk, ra)
    left_hip_angle = angle_3pt(ls, lh, lk)
    right_hip_angle = angle_3pt(rs, rh, rk)

    torso_vec = shoulder_mid[:2] - hip_mid[:2]
    vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    torso_lean = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(torso_vec, vertical_up)
                    / (safe_norm(torso_vec) * safe_norm(vertical_up)),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    hip_depth = float(hip_mid[1])
    knee_sep = float(abs(lk[0] - rk[0]))
    heel_sep = float(abs(lheel[0] - rheel[0]))

    return np.array(
        [
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            torso_lean,
            hip_depth,
            knee_sep,
            heel_sep,
        ],
        dtype=np.float32,
    )


def extract_frame_features(
    video_path: str,
    landmarker: vision.PoseLandmarker
) -> Tuple[Optional[np.ndarray], Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"status": "open_failed", "video_path": video_path}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_features = []
    total_frames_read = 0
    valid_pose_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames_read += 1
        lm = extract_landmarks_from_frame(frame, landmarker)
        if lm is None:
            continue

        valid_pose_frames += 1
        norm_lm = normalize_landmarks(lm)
        feats = compute_8_features(norm_lm)
        all_features.append(feats)

    cap.release()

    valid_ratio = valid_pose_frames / total_frames_read if total_frames_read > 0 else 0.0

    stats = {
        "status": "ok" if len(all_features) > 0 else "no_pose_detected",
        "video_path": video_path,
        "frame_count_meta": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_read": total_frames_read,
        "valid_pose_frames": valid_pose_frames,
        "valid_ratio": valid_ratio,
    }

    if len(all_features) == 0 or valid_ratio < MIN_VALID_RATIO:
        stats["status"] = "insufficient_valid_frames"
        return None, stats

    return np.stack(all_features, axis=0).astype(np.float32), stats


# ============================================================
# TASK 1: WHOLE-VIDEO DATASET
# ============================================================
def build_task1_dataset(
    metadata_csv: str = METADATA_CSV,
    model_path: str = MODEL_PATH,
    target_frames: int = TARGET_FRAMES,
):
    df = pd.read_csv(metadata_csv)

    required_cols = {"video_path", "file_name", "exercise_label", "dataset_source", "extension"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")

    df = df[df["exercise_label"].str.lower() == "squat"].copy()
    df["extension"] = df["extension"].str.lower()
    df = df[df["extension"].isin(VIDEO_EXTENSIONS)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No squat videos found in metadata CSV.")

    landmarker = get_pose_landmarker(model_path)

    X_list = []
    meta_rows = []
    failed_rows = []

    print(f"\nProcessing Task 1 on {len(df)} squat videos...")

    for idx, row in df.iterrows():
        video_path = row["video_path"]
        print(f"[{idx + 1}/{len(df)}] Task 1: {video_path}")

        try:
            features, stats = extract_frame_features(video_path, landmarker)

            if features is None:
                failed_rows.append({**row.to_dict(), **stats})
                print(f"  -> skipped ({stats['status']})")
                continue

            features_60 = resample_sequence(features, target_len=target_frames)
            X_list.append(features_60)

            meta_rows.append({**row.to_dict(), **stats})
            print(f"  -> saved shape {features_60.shape}")

        except Exception as e:
            failed_rows.append({**row.to_dict(), "status": f"exception: {str(e)}"})
            print(f"  -> exception: {e}")

    if len(X_list) == 0:
        raise RuntimeError("No videos were successfully processed for Task 1.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.zeros(len(X), dtype=np.int64)

    processed_df = pd.DataFrame(meta_rows)
    failed_df = pd.DataFrame(failed_rows)

    np.save(TASK1_OUTPUT_DIR / "X_squat_real.npy", X)
    np.save(TASK1_OUTPUT_DIR / "y_squat_real.npy", y)
    processed_df.to_csv(TASK1_OUTPUT_DIR / "processed_metadata.csv", index=False)
    failed_df.to_csv(TASK1_OUTPUT_DIR / "failed_metadata.csv", index=False)

    print("\nTask 1 complete.")
    print(f"Final X shape: {X.shape}")
    print(f"Saved: {TASK1_OUTPUT_DIR / 'X_squat_real.npy'}")
    print(f"Saved: {TASK1_OUTPUT_DIR / 'y_squat_real.npy'}")
    print(f"Saved: {TASK1_OUTPUT_DIR / 'processed_metadata.csv'}")
    print(f"Saved: {TASK1_OUTPUT_DIR / 'failed_metadata.csv'}")


# ============================================================
# TASK 2: REP SEGMENTATION
# ============================================================
def segment_squat_reps_from_features(
    features: np.ndarray,
    min_rep_frames: int = MIN_REP_FRAMES,
    max_rep_frames: int = MAX_REP_FRAMES,
    min_bottom_distance: int = MIN_BOTTOM_DISTANCE,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses average knee angle for squat rep segmentation.
    Squat bottom = minimum average knee angle.
    """
    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError("Expected features of shape [T, 8].")

    raw_signal = ((features[:, 0] + features[:, 1]) / 2.0).astype(np.float32)
    smooth_sig = smooth_signal(raw_signal)

    signal_std = float(np.std(smooth_sig))
    prominence = max(1.5, 0.14 * signal_std)

    bottoms, _ = find_peaks(
        -smooth_sig,
        distance=min_bottom_distance,
        prominence=prominence
    )

    rep_segments = []
    if len(bottoms) == 0:
        return rep_segments, raw_signal, smooth_sig, bottoms

    for i, bottom in enumerate(bottoms):
        if i == 0:
            left_center = max(0, bottom // 2)
            left_pad = max(TOP_SEARCH_PAD, bottom // 2)
            start = find_local_extreme(smooth_sig, left_center, left_pad, mode="max")
        else:
            prev_bottom = bottoms[i - 1]
            midpoint = (prev_bottom + bottom) // 2
            start = find_local_extreme(smooth_sig, midpoint, TOP_SEARCH_PAD, mode="max")

        if i == len(bottoms) - 1:
            right_center = bottom + max(1, (len(smooth_sig) - bottom) // 2)
            right_pad = max(TOP_SEARCH_PAD, (len(smooth_sig) - bottom) // 2)
            end = find_local_extreme(smooth_sig, right_center, right_pad, mode="max")
        else:
            next_bottom = bottoms[i + 1]
            midpoint = (bottom + next_bottom) // 2
            end = find_local_extreme(smooth_sig, midpoint, TOP_SEARCH_PAD, mode="max")

        if end <= start:
            continue

        seg_len = end - start + 1
        if seg_len < min_rep_frames or seg_len > max_rep_frames:
            continue

        if not (start < bottom < end):
            continue

        rep_segments.append((int(start), int(end)))

    cleaned = []
    for seg in rep_segments:
        if not cleaned:
            cleaned.append(seg)
            continue

        prev_s, prev_e = cleaned[-1]
        cur_s, cur_e = seg

        overlap = max(0, min(prev_e, cur_e) - max(prev_s, cur_s) + 1)
        prev_len = prev_e - prev_s + 1
        cur_len = cur_e - cur_s + 1

        if overlap / min(prev_len, cur_len) > 0.5:
            if cur_len > prev_len:
                cleaned[-1] = seg
        else:
            cleaned.append(seg)

    return cleaned, raw_signal, smooth_sig, bottoms


def build_task2_rep_dataset(
    metadata_csv: str = str(TASK1_OUTPUT_DIR / "processed_metadata.csv"),
    model_path: str = MODEL_PATH,
    target_frames: int = TARGET_FRAMES,
):
    df = pd.read_csv(metadata_csv)

    required = {"video_path", "file_name", "exercise_label", "dataset_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["exercise_label"].str.lower() == "squat"].reset_index(drop=True)
    if df.empty:
        raise ValueError("No squat rows found in metadata.")

    landmarker = get_pose_landmarker(model_path)

    rep_feature_list = []
    rep_rows = []
    failed_rows = []

    print(f"\nProcessing Task 2 on {len(df)} squat videos...")

    for idx, row in df.iterrows():
        video_path = row["video_path"]
        print(f"[{idx + 1}/{len(df)}] Task 2: {video_path}")

        try:
            features, stats = extract_frame_features(video_path, landmarker)
            if features is None:
                failed_rows.append({**row.to_dict(), **stats})
                print(f"  -> skipped ({stats['status']})")
                continue

            rep_segments, raw_signal, smooth_sig, bottoms = segment_squat_reps_from_features(features)

            print(f"  signal length = {len(raw_signal)}")
            print(f"  detected bottoms = {len(bottoms)}")
            if len(bottoms) > 0:
                print(f"  bottom indices = {bottoms[:10]}")

            if len(rep_segments) == 0:
                failed_rows.append({
                    **row.to_dict(),
                    **stats,
                    "status": "no_reps_detected",
                    "detected_bottoms": len(bottoms),
                })
                print("  -> no reps detected")
                continue

            print(f"  -> detected {len(rep_segments)} reps")

            for rep_idx, (start, end) in enumerate(rep_segments, start=1):
                rep_seq = features[start:end + 1]
                rep_seq_60 = resample_sequence(rep_seq, target_len=target_frames)

                rep_feature_list.append(rep_seq_60)
                rep_rows.append({
                    **row.to_dict(),
                    **stats,
                    "rep_index": rep_idx,
                    "rep_start_frame": start,
                    "rep_end_frame": end,
                    "rep_num_frames": end - start + 1,
                    "num_reps_in_video": len(rep_segments),
                    "detected_bottoms": len(bottoms),
                })

        except Exception as e:
            failed_rows.append({**row.to_dict(), "status": f"exception: {str(e)}"})
            print(f"  -> exception: {e}")

    if len(rep_feature_list) == 0:
        raise RuntimeError("No squat reps were successfully segmented.")

    X_reps = np.stack(rep_feature_list, axis=0).astype(np.float32)
    y_reps = np.zeros(len(X_reps), dtype=np.int64)

    rep_df = pd.DataFrame(rep_rows)
    failed_df = pd.DataFrame(failed_rows)

    np.save(TASK2_OUTPUT_DIR / "X_squat_reps.npy", X_reps)
    np.save(TASK2_OUTPUT_DIR / "y_squat_reps.npy", y_reps)
    rep_df.to_csv(TASK2_OUTPUT_DIR / "rep_metadata.csv", index=False)
    failed_df.to_csv(TASK2_OUTPUT_DIR / "rep_failed_metadata.csv", index=False)

    print("\nTask 2 complete.")
    print(f"Rep-level X shape: {X_reps.shape}")
    print(f"Saved: {TASK2_OUTPUT_DIR / 'X_squat_reps.npy'}")
    print(f"Saved: {TASK2_OUTPUT_DIR / 'y_squat_reps.npy'}")
    print(f"Saved: {TASK2_OUTPUT_DIR / 'rep_metadata.csv'}")
    print(f"Saved: {TASK2_OUTPUT_DIR / 'rep_failed_metadata.csv'}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Harsha squat pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["metadata", "task1", "task2", "all"],
        help="What to run"
    )
    args = parser.parse_args()

    if args.mode == "metadata":
        build_squat_metadata(SQUAT_FOLDERS, METADATA_CSV)

    elif args.mode == "task1":
        if not Path(METADATA_CSV).exists():
            build_squat_metadata(SQUAT_FOLDERS, METADATA_CSV)
        build_task1_dataset(METADATA_CSV, MODEL_PATH, TARGET_FRAMES)

    elif args.mode == "task2":
        processed_csv = TASK1_OUTPUT_DIR / "processed_metadata.csv"
        if not processed_csv.exists():
            raise FileNotFoundError(f"{processed_csv} not found. Run --mode task1 first.")
        build_task2_rep_dataset(str(processed_csv), MODEL_PATH, TARGET_FRAMES)

    elif args.mode == "all":
        build_squat_metadata(SQUAT_FOLDERS, METADATA_CSV)
        build_task1_dataset(METADATA_CSV, MODEL_PATH, TARGET_FRAMES)
        build_task2_rep_dataset(str(TASK1_OUTPUT_DIR / "processed_metadata.csv"), MODEL_PATH, TARGET_FRAMES)


if __name__ == "__main__":
    main()
"""
data/dataset_loader.py

Loads real squat videos and produces clips in the same dict format as
synthetic_loader.py, but with form_scores all set to -1.0 (unlabeled).

Sources
-------
  workoutfitness : data/workoutfitness-video/squat/
  similar        : data/real-time-exercise-recognition-dataset/similar_dataset/squat/
  final_kaggle   : data/real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat/
  my_test        : data/real-time-exercise-recognition-dataset/my_test_video_1/squat/
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pipeline import _extract_landmarks
from preprocessing.rep_segmenter import segment_reps
from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI"}

_SOURCE_MAP = {
    "workoutfitness-video":                        "workoutfitness",
    "similar_dataset":                             "similar",
    "final_kaggle_with_additional_video":          "final_kaggle",
    "my_test_video_1":                             "my_test",
}

_DEFAULT_DIRS = [
    "data/workoutfitness-video/squat",
    "data/real-time-exercise-recognition-dataset/similar_dataset/squat",
    "data/real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat",
    "data/real-time-exercise-recognition-dataset/my_test_video_1/squat",
]

_MIN_DETECTION_RATIO = 0.30   # skip if BlazePose detects pose in <30% of frames


def _source_name(folder: pathlib.Path) -> str:
    """Derive a short source label from the folder path."""
    for part in reversed(folder.parts):
        if part in _SOURCE_MAP:
            return _SOURCE_MAP[part]
    return folder.parts[-2] if len(folder.parts) >= 2 else folder.name


def _detection_ratio(landmarks: np.ndarray) -> float:
    """Fraction of frames where at least one hip landmark was detected."""
    detected = np.any(landmarks[:, 23:25, :3] != 0, axis=(1, 2))
    return float(detected.mean())


def load_real_squats(video_dirs: list[str] | None = None) -> list[dict]:
    """
    Extract landmarks and segment reps for every real squat video.

    Parameters
    ----------
    video_dirs : list of folder path strings, or None to use all 4 defaults.

    Returns
    -------
    list of dicts, one per successfully processed video:
        video_id    : str   filename without extension
        source      : str   workoutfitness / similar / final_kaggle / my_test
        landmarks   : np.ndarray [T, 33, 4]
        form_scores : np.ndarray [T]  all -1.0 (unlabeled)
        reps        : list of (start, end) tuples
        n_frames    : int
        n_reps      : int
        fps         : float
    """
    dirs = [pathlib.Path(d) for d in (video_dirs or _DEFAULT_DIRS)]

    clips    = []
    skipped  = {"not_found": 0, "open_error": 0, "low_detection": 0, "no_reps": 0}

    for folder in dirs:
        if not folder.exists():
            print(f"  [SKIP] folder not found: {folder}")
            skipped["not_found"] += 1
            continue

        source   = _source_name(folder)
        videos   = sorted(f for f in folder.iterdir() if f.suffix in VIDEO_EXTENSIONS)
        print(f"\n  Source '{source}': {len(videos)} video(s) in {folder}")

        for vpath in videos:
            print(f"    {vpath.name} ... ", end="", flush=True)

            try:
                landmarks = _extract_landmarks(str(vpath))   # [T, 33, 4]
            except Exception as e:
                print(f"ERROR ({e})")
                skipped["open_error"] += 1
                continue

            det_ratio = _detection_ratio(landmarks)
            if det_ratio < _MIN_DETECTION_RATIO:
                print(f"SKIP (detection {det_ratio:.0%} < 30%)")
                skipped["low_detection"] += 1
                continue

            reps = segment_reps(landmarks)

            import cv2
            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            T = landmarks.shape[0]
            print(f"OK  frames={T}  reps={len(reps)}  det={det_ratio:.0%}")

            clips.append({
                "video_id":    vpath.stem,
                "source":      source,
                "landmarks":   landmarks,
                "form_scores": np.full(T, -1.0, dtype=np.float32),
                "reps":        reps,
                "n_frames":    T,
                "n_reps":      len(reps),
                "fps":         fps,
            })

    print(f"\n=== dataset_loader ===")
    print(f"  Loaded  : {len(clips)} clips")
    print(f"  Skipped : {skipped}")
    print(f"  Total reps: {sum(c['n_reps'] for c in clips)}")
    return clips


def build_real_dataset(clips: list[dict]):
    """
    Preprocess real clips into fixed-length feature matrices.

    Parameters
    ----------
    clips : output of load_real_squats()

    Returns
    -------
    X        : np.ndarray [N, 60, 8]
    groups   : np.ndarray [N]  video index per rep
    metadata : list[dict] [N]  {video_id, source, rep_number, start, end}
    """
    X_list, groups_list, meta_list = [], [], []

    for video_idx, clip in enumerate(clips):
        norm_lm  = normalize(clip["landmarks"])          # [T, 33, 4]
        features = build_feature_matrix(norm_lm)        # [T, 8]

        for rep_num, (start, end) in enumerate(clip["reps"], start=1):
            rep_feat  = features[start:end + 1]         # [rep_T, 8]
            resampled = resample_to_60(rep_feat)        # [60, 8]

            X_list.append(resampled)
            groups_list.append(video_idx)
            meta_list.append({
                "video_id":   clip["video_id"],
                "source":     clip["source"],
                "rep_number": rep_num,
                "start":      start,
                "end":        end,
            })

    X      = np.stack(X_list)                    # [N, 60, 8]
    groups = np.array(groups_list, dtype=np.int32)

    return X, groups, meta_list


if __name__ == "__main__":
    from collections import defaultdict

    print("Loading real squat videos...\n")
    clips = load_real_squats()

    # ── Per-source summary ────────────────────────────────────────
    by_source = defaultdict(list)
    for c in clips:
        by_source[c["source"]].append(c)

    print("\nPer-source summary:")
    print(f"  {'Source':<16} {'Clips':>6} {'Reps':>6}  {'Avg fps':>8}")
    print("  " + "-" * 42)
    for src, src_clips in sorted(by_source.items()):
        total_reps = sum(c["n_reps"] for c in src_clips)
        avg_fps    = np.mean([c["fps"] for c in src_clips])
        print(f"  {src:<16} {len(src_clips):>6} {total_reps:>6}  {avg_fps:>8.1f}")

    total_reps = sum(c["n_reps"] for c in clips)
    print(f"\n  Total clips: {len(clips)}  Total reps: {total_reps}")

    # ── Build dataset and verify shape ────────────────────────────
    if clips:
        X, groups, metadata = build_real_dataset(clips)
        print(f"\n  X shape   : {X.shape}")
        assert X.shape[1:] == (60, 8), f"Expected [N, 60, 8], got {X.shape}"
        assert len(groups) == len(metadata) == X.shape[0]
        print(f"  groups    : {groups.shape}  unique videos: {len(set(groups.tolist()))}")
        print(f"  metadata  : {len(metadata)} entries")
        print("\n  Shape check PASS")

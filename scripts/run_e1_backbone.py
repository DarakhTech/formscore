"""
scripts/run_e1_backbone.py

E1 Backbone Comparison: BlazePose vs RTMPose on 20 real squat videos.

Metrics per backbone (averaged over 20 workoutfitness squat videos):
  detection_rate   — fraction of frames with a valid pose detection
  hip_signal_range — max-min of hip_mid_y (higher = cleaner squat signal)
  seg_f1           — F1 of rep count vs proxy expected (|det - exp| <= 1)
  feature_variance — std of build_feature_matrix [T, 8] (signal richness)
  avg_time_ms      — mean inference time per frame in milliseconds
"""

import sys
import pathlib
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pipeline import _extract_landmarks, _segment_reps
from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix

# ── Constants ──────────────────────────────────────────────────────────
RESULTS_DIR   = pathlib.Path("results")
SQUAT_DIR     = pathlib.Path("data/workoutfitness-video/squat")
N_VIDEOS      = 20
AVG_REP_DUR_S = 3.5   # proxy: avg squat rep duration
TOLERANCE     = 1     # |detected - expected| <= 1 → correct

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI"}

# COCO-17 index → MediaPipe-33 pipeline index
# Covers shoulders (5-6), elbows (7-8), wrists (9-10),
# hips (11-12), knees (13-14), ankles (15-16).
COCO_TO_PIPELINE = {
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}


# ── RTMPose extraction ──────────────────────────────────────────────────

def _extract_landmarks_rtmpose(video_path: str) -> tuple[np.ndarray, float]:
    """
    Run RTMPose (lightweight CPU) on video → [T, 33, 4] array.

    Coordinates are normalised to [0,1] by frame dimensions so they are
    on the same scale as BlazePose output.  Unmapped pipeline slots stay 0.
    visibility channel = keypoint confidence score.

    Returns
    -------
    landmarks : np.ndarray [T, 33, 4]
    avg_ms    : float  — mean per-frame inference time in milliseconds
    """
    from rtmlib import Body, PoseTracker

    tracker = PoseTracker(
        Body,
        mode="lightweight",
        backend="onnxruntime",
        device="cpu",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, round(source_fps / 30))

    all_landmarks: list[np.ndarray] = []
    frame_times:   list[float]      = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0             = time.perf_counter()
        keypoints, scores = tracker(rgb)
        frame_times.append((time.perf_counter() - t0) * 1000)

        lm_array = np.zeros((33, 4), dtype=np.float32)

        if keypoints is not None and len(keypoints) > 0:
            kps = keypoints[0]   # [17, 2]  pixel coords
            sc  = scores[0]      # [17]

            for coco_idx, pipe_idx in COCO_TO_PIPELINE.items():
                if coco_idx < kps.shape[0]:
                    lm_array[pipe_idx, 0] = kps[coco_idx, 0] / w   # x norm
                    lm_array[pipe_idx, 1] = kps[coco_idx, 1] / h   # y norm
                    lm_array[pipe_idx, 2] = 0.0
                    lm_array[pipe_idx, 3] = float(sc[coco_idx])

        all_landmarks.append(lm_array)

    cap.release()

    landmarks = (
        np.stack(all_landmarks)
        if all_landmarks
        else np.zeros((0, 33, 4), dtype=np.float32)
    )
    avg_ms = float(np.mean(frame_times)) if frame_times else 0.0
    return landmarks, avg_ms


# ── Per-video metrics ───────────────────────────────────────────────────

def _video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _per_video_metrics(landmarks: np.ndarray, fps: float) -> dict:
    """
    Compute the four benchmark metrics from a [T, 33, 4] landmark array.

    Hip landmarks are at pipeline indices 23 (left) and 24 (right).
    """
    T = landmarks.shape[0]

    # 1. detection_rate — at least one hip landmark non-zero
    detected       = np.any(landmarks[:, 23:25, :2] != 0, axis=(1, 2))
    detection_rate = float(detected.mean())

    # 2. hip_signal_range — range of mid-hip Y across the video
    hip_mid_y       = (landmarks[:, 23, 1] + landmarks[:, 24, 1]) / 2.0
    hip_signal_range = float(hip_mid_y.max() - hip_mid_y.min())

    # 3. seg_f1 (per-video: 1.0 if within tolerance, 0.0 otherwise)
    duration_s    = T / fps
    expected_reps = round(duration_s / AVG_REP_DUR_S)
    reps          = _segment_reps(landmarks, exercise="squat")
    detected_reps = len(reps)
    correct       = abs(detected_reps - expected_reps) <= TOLERANCE

    # 4. feature_variance — std of the full [T, 8] feature matrix
    try:
        norm_lm          = normalize(landmarks)
        feat             = build_feature_matrix(norm_lm, exercise="squat")   # [T, 8]
        feature_variance = float(np.std(feat))
    except Exception:
        feature_variance = 0.0

    return {
        "detection_rate":  detection_rate,
        "hip_signal_range": hip_signal_range,
        "seg_correct":      correct,
        "detected_reps":    detected_reps,
        "expected_reps":    expected_reps,
        "feature_variance": feature_variance,
    }


def _aggregate_f1(rows: list[dict]) -> float:
    """
    Compute F1 over a collection of per-video result dicts.

    Follows the same formula used in E2:
      precision = detected_reps_in_correct_videos / total_detected_reps
      recall    = correct_videos / total_videos
    """
    total       = len(rows)
    total_det   = sum(r["detected_reps"] for r in rows)
    n_correct   = sum(int(r["seg_correct"]) for r in rows)
    det_correct = sum(r["detected_reps"] for r in rows if r["seg_correct"])

    precision = det_correct / total_det if total_det > 0 else 0.0
    recall    = n_correct   / total     if total     > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    videos = sorted(
        [f for f in SQUAT_DIR.iterdir() if f.suffix in VIDEO_EXTS],
        key=lambda p: p.stem,
    )[:N_VIDEOS]

    if not videos:
        print(f"No videos found in {SQUAT_DIR}")
        return

    print(f"E1 Backbone Comparison — {len(videos)} videos from {SQUAT_DIR}\n")

    bp_rows:  list[dict] = []
    rtm_rows: list[dict] = []
    csv_rows: list[dict] = []

    bp_times:  list[float] = []
    rtm_times: list[float] = []

    for i, vpath in enumerate(videos, start=1):
        fps = _video_fps(str(vpath))
        print(f"[{i:>2}/{len(videos)}] {vpath.name}")

        # ── BlazePose ─────────────────────────────────────────
        print("      BlazePose  ... ", end="", flush=True)
        t0   = time.perf_counter()
        lm_bp = _extract_landmarks(str(vpath))
        bp_elapsed_ms = (time.perf_counter() - t0) * 1000
        bp_frame_ms   = bp_elapsed_ms / max(len(lm_bp), 1)
        bp_times.append(bp_frame_ms)

        bp_m = _per_video_metrics(lm_bp, fps)
        bp_rows.append(bp_m)
        print(
            f"det={bp_m['detection_rate']:.2f}  "
            f"range={bp_m['hip_signal_range']:.3f}  "
            f"reps={bp_m['detected_reps']}/{bp_m['expected_reps']}  "
            f"{bp_frame_ms:.1f}ms/frame"
        )

        # ── RTMPose ───────────────────────────────────────────
        print("      RTMPose    ... ", end="", flush=True)
        lm_rtm, rtm_frame_ms = _extract_landmarks_rtmpose(str(vpath))
        rtm_times.append(rtm_frame_ms)

        rtm_m = _per_video_metrics(lm_rtm, fps)
        rtm_rows.append(rtm_m)
        print(
            f"det={rtm_m['detection_rate']:.2f}  "
            f"range={rtm_m['hip_signal_range']:.3f}  "
            f"reps={rtm_m['detected_reps']}/{rtm_m['expected_reps']}  "
            f"{rtm_frame_ms:.1f}ms/frame"
        )

        csv_rows.append({
            "video":              vpath.name,
            "bp_detection_rate":  bp_m["detection_rate"],
            "bp_hip_range":       bp_m["hip_signal_range"],
            "bp_seg_correct":     int(bp_m["seg_correct"]),
            "bp_det_reps":        bp_m["detected_reps"],
            "bp_exp_reps":        bp_m["expected_reps"],
            "bp_feat_variance":   bp_m["feature_variance"],
            "bp_ms_per_frame":    bp_frame_ms,
            "rtm_detection_rate": rtm_m["detection_rate"],
            "rtm_hip_range":      rtm_m["hip_signal_range"],
            "rtm_seg_correct":    int(rtm_m["seg_correct"]),
            "rtm_det_reps":       rtm_m["detected_reps"],
            "rtm_exp_reps":       rtm_m["expected_reps"],
            "rtm_feat_variance":  rtm_m["feature_variance"],
            "rtm_ms_per_frame":   rtm_frame_ms,
        })

    # ── Aggregate ────────────────────────────────────────────────────
    bp_det  = np.mean([r["detection_rate"]  for r in bp_rows])
    bp_hip  = np.mean([r["hip_signal_range"] for r in bp_rows])
    bp_f1   = _aggregate_f1(bp_rows)
    bp_var  = np.mean([r["feature_variance"] for r in bp_rows])
    bp_ms   = np.mean(bp_times)

    rtm_det = np.mean([r["detection_rate"]  for r in rtm_rows])
    rtm_hip = np.mean([r["hip_signal_range"] for r in rtm_rows])
    rtm_f1  = _aggregate_f1(rtm_rows)
    rtm_var = np.mean([r["feature_variance"] for r in rtm_rows])
    rtm_ms  = np.mean(rtm_times)

    # ── Print comparison table ────────────────────────────────────────
    rule = "─" * 55
    print(f"\n\n  E1 Backbone Comparison  ({len(videos)} squat videos)")
    print(f"  {rule}")
    print(f"  {'Metric':<22} {'BlazePose':>12} {'RTMPose':>12}")
    print(f"  {rule}")
    print(f"  {'Detection rate':<22} {bp_det:>12.3f} {rtm_det:>12.3f}")
    print(f"  {'Hip signal range':<22} {bp_hip:>12.3f} {rtm_hip:>12.3f}")
    print(f"  {'Segmenter F1':<22} {bp_f1:>12.3f} {rtm_f1:>12.3f}")
    print(f"  {'Feature variance':<22} {bp_var:>12.3f} {rtm_var:>12.3f}")
    print(f"  {'Avg time/frame ms':<22} {bp_ms:>12.1f} {rtm_ms:>12.1f}")
    print(f"  {rule}")

    # ── Save CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame(csv_rows)
    out_path = RESULTS_DIR / "E1_backbone.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()

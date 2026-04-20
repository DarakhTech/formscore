"""
pipeline.py

FormScore end-to-end pipeline — Stage 1 through 5.

Input  : path to an .mp4 file
Output : dict (and optionally saved JSON) with per-rep scores,
         SHAP fault analysis, and feedback cues.

Usage:
    from pipeline import FormScorePipeline

    pipeline = FormScorePipeline(model_fn=my_model, background=bg_data)
    result   = pipeline.run("squat.mp4", save_json="results/out.json")

Output schema:
    {
        "video":       str,
        "exercise":    str,
        "n_reps":      int,
        "reps": [
            {
                "rep_number":   int,
                "start_frame":  int,
                "end_frame":    int,
                "form_score":   float,
                "feedback":     {
                    "overall":    str,
                    "cues":       [str],
                    "top_fault":  str,
                    "frame_peak": int,
                },
                "shap_values":  [[60 x 8]],   # only if include_shap=True
            },
            ...
        ],
        "summary": {
            "mean_score":   float,
            "best_rep":     int,
            "worst_rep":    int,
            "latency_ms":   float,   # per-rep average
        }
    }
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarkerOptions, PoseLandmarker, RunningMode
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60
from explainability.shap_explainer import FormScoreExplainer, FEATURE_NAMES
from explainability.feedback_lookup import get_feedback

# ── MediaPipe model ───────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

def _download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(url, MODEL_PATH)


# ── Stage 1: Pose extraction ──────────────────────────────

def _extract_landmarks(video_path: str) -> np.ndarray:
    """
    Run BlazePose on video → [T, 33, 4] landmark array.
    Frames with no detection are filled with zeros.
    """
    _download_model()
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

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    all_landmarks = []
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
            )   # [33, 4]
        else:
            lm_array = np.zeros((33, 4), dtype=np.float32)

        all_landmarks.append(lm_array)

    cap.release()
    landmarker.close()

    return np.stack(all_landmarks)   # [T, 33, 4]


# ── Stage 2: Rep segmentation ─────────────────────────────

def _segment_reps(landmarks_raw: np.ndarray,
                  smooth_sigma: float = 3.0,
                  min_distance: int = 30) -> list[tuple[int, int]]:
    """
    Detect rep boundaries from RAW (unnormalized) landmarks.
    Hip y in image coords increases downward — squat bottom = peak.
    """
    hip_mid_y = (landmarks_raw[:, 23, 1] + landmarks_raw[:, 24, 1]) / 2.0
    smoothed  = gaussian_filter1d(hip_mid_y, sigma=smooth_sigma)

    signal_range = smoothed.max() - smoothed.min()
    prominence   = signal_range * 0.30   # 30% of range

    bottoms, _ = find_peaks( smoothed, distance=min_distance, prominence=prominence)
    tops,    _ = find_peaks(-smoothed, distance=min_distance, prominence=prominence)

    if len(bottoms) == 0:
        return [(0, len(hip_mid_y) - 1)]
    if len(tops) == 0:
        return [(0, len(hip_mid_y) - 1)]

    reps = []
    for bottom in bottoms:
        before = tops[tops < bottom]
        after  = tops[tops > bottom]
        start  = int(before[-1]) if len(before) > 0 else 0
        end    = int(after[0])   if len(after)  > 0 else len(hip_mid_y) - 1
        if end > start + 10:
            reps.append((start, end))

    if not reps:
        return [(0, len(hip_mid_y) - 1)]

    reps   = sorted(set(reps))
    merged = [reps[0]]
    for r in reps[1:]:
        if r[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged

# ── Stub scorer (replaced by real model in Sprint 3) ─────

def _stub_model(X: np.ndarray) -> np.ndarray:
    """
    Stub scorer for pipeline testing without a trained model.
    Penalizes knee asymmetry and spine tilt.
    """
    symmetry = X[:, :, 7].mean(axis=1)
    spine    = X[:, :, 4].mean(axis=1) / 180.0
    score    = 1.0 - 0.4 * symmetry - 0.3 * spine
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# ── Main pipeline class ───────────────────────────────────

class FormScorePipeline:
    """
    End-to-end FormScore pipeline.

    Parameters
    ----------
    model_fn   : callable [N, 60, 8] → [N]
                 Scoring model. Defaults to stub until real model exists.
    background : np.ndarray [B, 60, 8]
                 SHAP background dataset. Defaults to random if None.
    exercise   : str
                 Exercise type label for output metadata.
    """

    def __init__(self,
                 model_fn=None,
                 background: np.ndarray = None,
                 exercise: str = "squat"):

        self.model_fn  = model_fn or _stub_model
        self.exercise  = exercise

        # Build SHAP explainer
        if background is None:
            # Random background until real training data is passed in
            rng        = np.random.default_rng(42)
            background = rng.random((50, 60, 8)).astype(np.float32)

        self.explainer = FormScoreExplainer(
            self.model_fn, background, model_type="kernel"
        )

    def run(self,
            video_path: str,
            save_json: str = None,
            include_shap: bool = False) -> dict:
        """
        Run the full pipeline on a video file.

        Parameters
        ----------
        video_path   : path to .mp4
        save_json    : optional path to save output JSON
        include_shap : whether to include full [60,8] SHAP matrix in output

        Returns
        -------
        dict — full per-rep results and summary
        """
        t_total_start = time.perf_counter()

        # ── Stage 1: Pose extraction ──────────────────────
        landmarks_raw  = _extract_landmarks(video_path)      # [T, 33, 4]

        # ── Stage 2: Normalization ────────────────────────
        landmarks_norm = normalize(landmarks_raw)           # [T, 33, 4]

        # ── Stage 3: Rep segmentation ─────────────────────
        reps = _segment_reps(landmarks_raw)

        rep_results  = []
        latencies_ms = []

        for i, (start, end) in enumerate(reps):
            t_rep_start = time.perf_counter()

            rep_lm = landmarks_norm[start:end]   # [rep_T, 33, 4]

            if len(rep_lm) < 5:
                continue

            # ── Stage 4: Feature engineering ─────────────
            feat    = build_feature_matrix(rep_lm)   # [rep_T, 8]
            feat_60 = resample_to_60(feat)           # [60, 8]

            # ── Stage 5a: Scoring ─────────────────────────
            score = float(self.model_fn(feat_60[np.newaxis])[0])

            # ── Stage 5b: SHAP explanation ────────────────
            explanation = self.explainer.explain(feat_60)

            # ── Stage 5c: Feedback ────────────────────────
            feedback = get_feedback(explanation, form_score=score)

            t_rep_ms = (time.perf_counter() - t_rep_start) * 1000
            latencies_ms.append(t_rep_ms)

            rep_result = {
                "rep_number":  i + 1,
                "start_frame": start,
                "end_frame":   end,
                "form_score":  round(score, 3),
                "feedback":    {
                    "overall":    feedback["overall"],
                    "cues":       feedback["cues"],
                    "top_fault":  feedback["top_fault"],
                    "frame_peak": feedback["frame_peak"],
                },
            }

            if include_shap:
                rep_result["shap_values"] = explanation["shap_values"].tolist()

            rep_results.append(rep_result)

        # ── Summary ───────────────────────────────────────
        scores = [r["form_score"] for r in rep_results]
        output = {
            "video":    os.path.basename(video_path),
            "exercise": self.exercise,
            "n_reps":   len(rep_results),
            "reps":     rep_results,
            "summary": {
                "mean_score":  round(float(np.mean(scores)), 3) if scores else 0.0,
                "best_rep":    int(np.argmax(scores)) + 1       if scores else 0,
                "worst_rep":   int(np.argmin(scores)) + 1       if scores else 0,
                "latency_ms":  round(float(np.mean(latencies_ms)), 1) if latencies_ms else 0.0,
            }
        }

        total_ms = (time.perf_counter() - t_total_start) * 1000
        print(f"\n=== FormScore Pipeline ===")
        print(f"  Video    : {os.path.basename(video_path)}")
        print(f"  Reps     : {len(rep_results)}")
        print(f"  Scores   : {[r['form_score'] for r in rep_results]}")
        print(f"  Avg score: {output['summary']['mean_score']:.3f}")
        print(f"  Latency  : {output['summary']['latency_ms']:.1f}ms/rep")
        print(f"  Total    : {total_ms:.0f}ms")

        if save_json:
            os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
            with open(save_json, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Saved    : {save_json}")

        return output
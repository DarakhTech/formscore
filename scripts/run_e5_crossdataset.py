"""
scripts/run_e5_crossdataset.py

E5 Cross-Dataset Generalization:
  Train distribution : synthetic mocap (720 reps, scores 0.518-0.925)
  Test distribution  : real workout videos (data/workoutfitness-video/squat/)

Runs the full inference pipeline (BlazePose → normalize → segment →
features → resample → BiLSTM) on every real squat video, then compares
score distributions between synthetic and real data.
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pipeline import _extract_landmarks, _segment_reps
from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60
from modeling.load_model import predict_fn
from modeling.evaluate import _build_dataset

REAL_VIDEO_DIR = pathlib.Path("data/workoutfitness-video/squat")
RESULTS_DIR    = pathlib.Path("results")
MIN_REP_FRAMES = 10


def _synthetic_score_stats() -> dict:
    """Return mean/std/min/max of labels from the synthetic training set."""
    _, y, _ = _build_dataset()
    return {
        "mean": float(y.mean()),
        "std":  float(y.std()),
        "min":  float(y.min()),
        "max":  float(y.max()),
        "n":    len(y),
    }


def _process_video(video_path: pathlib.Path) -> list[dict]:
    """
    Run full pipeline on one video.
    Returns list of {rep_number, predicted_score} dicts.
    Skips silently on detection failures.
    """
    try:
        landmarks_raw  = _extract_landmarks(str(video_path))  # [T, 33, 4]
    except Exception as e:
        print(f"    [SKIP] {video_path.name}: landmark extraction failed — {e}")
        return []

    if landmarks_raw.shape[0] < MIN_REP_FRAMES:
        print(f"    [SKIP] {video_path.name}: too short ({landmarks_raw.shape[0]} frames)")
        return []

    landmarks_norm = normalize(landmarks_raw)                  # [T, 33, 4]
    reps           = _segment_reps(landmarks_raw)

    rep_records = []
    for i, (start, end) in enumerate(reps):
        rep_lm = landmarks_norm[start:end]
        if len(rep_lm) < MIN_REP_FRAMES:
            continue

        feat    = build_feature_matrix(rep_lm)    # [rep_T, 8]
        feat_60 = resample_to_60(feat)            # [60, 8]
        score   = float(predict_fn(feat_60[np.newaxis])[0])

        rep_records.append({"rep_number": i + 1, "predicted_score": round(score, 4)})

    return rep_records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Synthetic distribution ────────────────────────────────────
    print("Loading synthetic score distribution...")
    synth = _synthetic_score_stats()
    print(f"  Synthetic: {synth['n']} reps  mean={synth['mean']:.3f}  std={synth['std']:.3f}\n")

    # ── Real video inference ──────────────────────────────────────
    video_files = sorted(
        REAL_VIDEO_DIR.glob("*"),
        key=lambda p: p.stem,
    )
    video_files = [v for v in video_files if v.suffix.lower() in {".mp4", ".mov", ".avi"}]

    print(f"Found {len(video_files)} real squat videos in {REAL_VIDEO_DIR}\n")

    per_rep_rows = []
    for video_path in video_files:
        print(f"  Processing: {video_path.name}")
        reps = _process_video(video_path)
        if reps:
            scores = [r["predicted_score"] for r in reps]
            print(f"    {len(reps)} rep(s)  scores: {[round(s, 3) for s in scores]}")
            for r in reps:
                per_rep_rows.append({
                    "video_filename":  video_path.name,
                    "rep_number":      r["rep_number"],
                    "predicted_score": r["predicted_score"],
                })
        else:
            print(f"    0 reps scored")

    # ── Real distribution stats ───────────────────────────────────
    real_scores = np.array([r["predicted_score"] for r in per_rep_rows], dtype=np.float32)

    if len(real_scores) == 0:
        print("\nNo reps successfully scored on real videos.")
        return

    real = {
        "mean": float(real_scores.mean()),
        "std":  float(real_scores.std()),
        "min":  float(real_scores.min()),
        "max":  float(real_scores.max()),
        "n":    len(real_scores),
    }

    domain_gap = abs(synth["mean"] - real["mean"])
    score_drop = ((synth["mean"] - real["mean"]) / synth["mean"]) * 100

    # ── Print summary ─────────────────────────────────────────────
    rule = "─" * 45
    print(f"\n  E5 Cross-Dataset Generalization")
    print(f"  {rule}")
    print(f"  Synthetic (train): mean={synth['mean']:.3f}  std={synth['std']:.3f}  "
          f"min={synth['min']:.3f}  max={synth['max']:.3f}")
    print(f"  Real videos (test): mean={real['mean']:.3f}  std={real['std']:.3f}  "
          f"min={real['min']:.3f}  max={real['max']:.3f}")
    print(f"  {rule}")
    print(f"  Domain gap (mean diff): {domain_gap:.3f}")
    sign = "↓" if score_drop > 0 else "↑"
    print(f"  Score drop: {abs(score_drop):.1f}% {sign}")
    print(f"  Reps scored on real videos: {real['n']}")
    print()

    # ── Save per-rep CSV ──────────────────────────────────────────
    per_rep_path = RESULTS_DIR / "E5_crossdataset.csv"
    pd.DataFrame(per_rep_rows).to_csv(per_rep_path, index=False)
    print(f"Saved → {per_rep_path}")

    # ── Save summary CSV ──────────────────────────────────────────
    summary_rows = [
        {"dataset": "synthetic", "mean_score": synth["mean"],
         "std_score": synth["std"], "n_reps": synth["n"]},
        {"dataset": "real",      "mean_score": real["mean"],
         "std_score": real["std"],  "n_reps": real["n"]},
    ]
    summary_path = RESULTS_DIR / "E5_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()

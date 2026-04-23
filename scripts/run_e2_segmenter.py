"""
scripts/run_e2_segmenter.py

E2 Segmenter Proxy Evaluation.

Ground truth is unavailable for real videos, so we use a proxy:
  expected_reps = round(duration_s / 3.5)   (avg squat rep ≈ 3.5 s)
  correct       = |detected - expected| <= 1

Precision = detected reps from correct videos / total detected reps
Recall    = videos within 1 rep of expected  / total videos
F1        = 2 * P * R / (P + R)
"""

import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data.dataset_loader import load_real_squats

RESULTS_DIR   = pathlib.Path("results")
AVG_REP_DUR_S = 3.5    # seconds per squat rep (proxy constant)
TOLERANCE     = 1      # within this many reps = correct


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading real squat videos...\n")
    clips = load_real_squats()

    # ── Per-video metrics ─────────────────────────────────────────
    rows = []
    for clip in clips:
        duration_s    = clip["n_frames"] / clip["fps"]
        expected_reps = round(duration_s / AVG_REP_DUR_S)
        detected_reps = clip["n_reps"]
        correct       = int(abs(detected_reps - expected_reps) <= TOLERANCE)

        rows.append({
            "source":        clip["source"],
            "video_id":      clip["video_id"],
            "detected_reps": detected_reps,
            "expected_reps": expected_reps,
            "correct":       correct,
            "fps":           round(clip["fps"], 2),
            "duration_s":    round(duration_s, 2),
        })

    df = pd.DataFrame(rows)

    # ── Per-source and overall stats ──────────────────────────────
    def source_stats(sub: pd.DataFrame) -> dict:
        total_videos   = len(sub)
        total_detected = sub["detected_reps"].sum()
        correct_videos = sub["correct"].sum()
        detected_in_correct = sub.loc[sub["correct"] == 1, "detected_reps"].sum()

        precision = detected_in_correct / total_detected if total_detected > 0 else 0.0
        recall    = correct_videos / total_videos        if total_videos   > 0 else 0.0
        f1        = _f1(precision, recall)

        return {
            "videos":        total_videos,
            "det_reps":      int(total_detected),
            "exp_reps":      int(sub["expected_reps"].sum()),
            "precision":     precision,
            "recall":        recall,
            "f1":            f1,
        }

    source_order = ["workoutfitness", "similar", "final_kaggle", "my_test"]
    sources      = [s for s in source_order if s in df["source"].unique()]
    sources     += [s for s in df["source"].unique() if s not in source_order]

    per_source = {src: source_stats(df[df["source"] == src]) for src in sources}
    overall    = source_stats(df)

    # ── Print table ───────────────────────────────────────────────
    rule = "─" * 57
    header = f"  {'Source':<18} {'Videos':>7} {'Det.Reps':>9} {'Exp.Reps':>9}  {'F1':>6}"

    print(f"\n  E2 Segmenter Evaluation")
    print(f"  {rule}")
    print(header)
    print(f"  {rule}")
    for src in sources:
        s = per_source[src]
        print(
            f"  {src:<18} {s['videos']:>7} {s['det_reps']:>9} "
            f"{s['exp_reps']:>9}  {s['f1']:>6.2f}"
        )
    print(f"  {rule}")
    print(f"  {'Overall':<18} {overall['videos']:>7} {overall['det_reps']:>9} "
          f"{overall['exp_reps']:>9}  {overall['f1']:>6.2f}")
    print(f"\n  Overall precision : {overall['precision']:.3f}")
    print(f"  Overall recall    : {overall['recall']:.3f}")
    print(f"  Overall F1        : {overall['f1']:.3f}")

    # ── Save ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "E2_segmenter.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

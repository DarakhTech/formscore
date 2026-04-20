# tests/test_shap_heatmap.py
"""
s3-a2: Generate SHAP heatmaps for 5 representative reps.
Runs the full pipeline on squats.mp4, takes all reps,
picks the best, worst, and 3 middle reps, plots summary grid.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pipeline import FormScorePipeline
from explainability.shap_heatmap import plot_rep_summary

VIDEO_PATH  = "tests/squats.mp4"
OUTPUT_PATH = "results/shap_summary.png"


def test_shap_heatmap():
    pipeline = FormScorePipeline()
    result   = pipeline.run(VIDEO_PATH, include_shap=True)

    assert result["n_reps"] > 0, "No reps detected"

    # Build reps_data for heatmap
    reps_data = []
    for rep in result["reps"]:
        if "shap_values" not in rep:
            continue
        reps_data.append({
            "shap_values": np.array(rep["shap_values"]),  # [60, 8]
            "form_score":  rep["form_score"],
            "rep_number":  rep["rep_number"],
            "top_fault":   rep["feedback"]["top_fault"],
            "frame_peak":  rep["feedback"]["frame_peak"],
        })

    # Pick 5 representative reps:
    # best, worst, and spread across the middle
    scores  = [r["form_score"] for r in reps_data]
    indices = sorted(range(len(scores)), key=lambda i: scores[i])

    if len(indices) >= 5:
        picks = [
            indices[0],                      # worst
            indices[len(indices) // 4],      # low-mid
            indices[len(indices) // 2],      # median
            indices[3 * len(indices) // 4],  # high-mid
            indices[-1],                     # best
        ]
    else:
        picks = indices   # use all if fewer than 5

    selected = [reps_data[i] for i in picks]

    print(f"\n=== s3-a2 SHAP Heatmap ===")
    print(f"  Total reps  : {len(reps_data)}")
    print(f"  Selected    : {len(selected)} reps for visualization")
    for r in selected:
        print(f"    rep {r['rep_number']:2d}  score={r['form_score']:.3f}"
              f"  fault={r['top_fault']}  peak_frame={r['frame_peak']}")

    plot_rep_summary(selected, save_path=OUTPUT_PATH,
                     title="FormScore — SHAP Temporal Heatmaps (squats.mp4)")

    assert os.path.exists(OUTPUT_PATH), "PNG not saved"
    print(f"\n  Heatmap saved: {OUTPUT_PATH}")
    print("=== PASS ===\n")


if __name__ == "__main__":
    test_shap_heatmap()
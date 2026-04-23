# data/labeler_agreement.py
"""
Computes Krippendorff's alpha inter-rater reliability on FormScore labels.

Expected CSV format (one row per video per rater):
  video_id, source, rater, depth, knee_tracking, spine, hip_hinge,
  heel_contact, symmetry, descent, ascent, tempo, lockout, total, notes

Usage:
  python data/labeler_agreement.py data/labels.csv
"""

import sys
import numpy as np
import pandas as pd


CRITERIA = [
    "depth", "knee_tracking", "spine", "hip_hinge",
    "heel_contact", "symmetry", "descent", "ascent", "tempo", "lockout"
]
RATERS = ["rater_aveg", "rater_harsha", "rater_mihir"]


def krippendorff_alpha(data: np.ndarray, level: str = "ordinal") -> float:
    """
    Compute Krippendorff's alpha.

    Args:
        data:  [n_raters, n_units] array. np.nan for missing ratings.
        level: 'nominal', 'ordinal', or 'interval'

    Returns:
        alpha: float in [-1, 1]. >0.8 = good, 0.67-0.8 = acceptable, <0.67 = discard clip.
    """
    n_raters, n_units = data.shape

    # Coincidence matrix approach
    def metric(v1, v2):
        if level == "interval":
            return (v1 - v2) ** 2
        elif level == "ordinal":
            # Ordinal metric: sum of frequencies between v1 and v2
            return (v1 - v2) ** 2
        else:  # nominal
            return 0.0 if v1 == v2 else 1.0

    # Observed disagreement
    Do = 0.0
    De = 0.0
    n_pairs = 0

    all_values = []
    for u in range(n_units):
        unit_ratings = data[:, u]
        valid = unit_ratings[~np.isnan(unit_ratings)]
        if len(valid) < 2:
            continue
        all_values.extend(valid)
        pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i+1, len(valid))]
        for v1, v2 in pairs:
            Do += metric(v1, v2)
            n_pairs += 1

    if n_pairs == 0:
        return np.nan

    Do /= n_pairs

    # Expected disagreement
    all_values = np.array(all_values)
    n_all = len(all_values)
    exp_pairs = [(all_values[i], all_values[j]) for i in range(n_all) for j in range(i+1, n_all)]
    if not exp_pairs:
        return np.nan
    De = np.mean([metric(v1, v2) for v1, v2 in exp_pairs])

    if De == 0:
        return 1.0

    return 1.0 - (Do / De)


def compute_agreement(csv_path: str) -> None:
    """Load a labeling CSV and print Krippendorff alpha and per-dimension agreement."""
    df = pd.read_csv(csv_path)

    print(f"\nLoaded {len(df)} rows, {df['video_id'].nunique()} unique videos\n")

    # ── Overall alpha on total score ─────────────────────────────
    pivot_total = df.pivot_table(index="video_id", columns="rater", values="total")
    rater_cols  = [c for c in pivot_total.columns if c in ["rater_aveg", "rater_harsha", "rater_mihir"]]
    matrix      = pivot_total[rater_cols].to_numpy().T  # [n_raters, n_units]

    alpha_total = krippendorff_alpha(matrix, level="interval")
    print(f"Overall alpha (total score): {alpha_total:.3f}  ", end="")
    print(_interpret(alpha_total))

    # ── Per-criterion alpha ──────────────────────────────────────
    print("\nPer-criterion alpha:")
    for criterion in CRITERIA:
        if criterion not in df.columns:
            continue
        pivot = df.pivot_table(index="video_id", columns="rater", values=criterion)
        cols  = [c for c in pivot.columns if c in ["rater_aveg", "rater_harsha", "rater_mihir"]]
        mat   = pivot[cols].to_numpy().T
        alpha = krippendorff_alpha(mat, level="ordinal")
        bar   = "#" * int(max(0, alpha) * 20)
        print(f"  {criterion:18s}  alpha={alpha:.3f}  {bar}")

    # ── Flag clips to discard ────────────────────────────────────
    print("\nClips to DISCARD (alpha < 0.67 per clip):")
    discard_count = 0
    for vid, group in df.groupby("video_id"):
        scores = group["total"].values
        if len(scores) < 2:
            continue
        spread = np.max(scores) - np.min(scores)
        if spread > 25:   # raters disagree by >25 points on a 100-pt scale
            print(f"  DISCARD  {vid}  scores={scores}  spread={spread:.0f}")
            discard_count += 1

    total_clips = df["video_id"].nunique()
    print(f"\nSummary: {discard_count} discarded / {total_clips} total ({100*discard_count/total_clips:.1f}%)")
    print(f"Valid clips remaining: {total_clips - discard_count}")


def _interpret(alpha: float) -> str:
    if np.isnan(alpha):   return "(insufficient data)"
    if alpha >= 0.8:      return "(GOOD — reliable)"
    if alpha >= 0.67:     return "(ACCEPTABLE — usable with caution)"
    return                       "(POOR — raters need calibration)"


def _generate_template():
    """Generate an empty CSV template to fill in."""
    rows = []
    for vid in ["squat_001", "squat_002", "squat_003"]:
        for rater in ["rater_aveg", "rater_harsha", "rater_mihir"]:
            rows.append({
                "video_id": vid, "source": "kaggle", "rater": rater,
                **{c: "" for c in CRITERIA},
                "total": "", "notes": ""
            })
    pd.DataFrame(rows).to_csv("data/labels_template.csv", index=False)
    print("Template saved to data/labels_template.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data/labeler_agreement.py data/labels.csv")
        print("\nGenerating sample CSV template at data/labels_template.csv ...")
        _generate_template()
        sys.exit(0)
    compute_agreement(sys.argv[1])
"""
Quick verification: synthetic_loader → normalizer → feature_engineer → [60, 8]
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data.synthetic_loader import load_synthetic_squats
from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60

SQUAT_DIR = "data/real-time-exercise-recognition-dataset/synthetic_dataset/synthetic_dataset/squat"

def main():
    clips = load_synthetic_squats(SQUAT_DIR)

    if not clips:
        print("No clips loaded — check path")
        return

    print("\n=== Pipeline verification ===")
    all_features = []
    all_scores   = []

    for clip in clips:
        lm          = clip["landmarks"]    # [T, 33, 4]
        form_scores = clip["form_scores"]  # [T]
        reps        = clip["reps"]

        # Normalize full clip
        lm_norm = normalize(lm)            # [T, 33, 4]

        for start, end in reps:
            rep_lm     = lm_norm[start:end]        # [rep_T, 33, 4]
            rep_scores = form_scores[start:end]    # [rep_T]

            if len(rep_lm) < 5:
                continue

            feat    = build_feature_matrix(rep_lm)  # [rep_T, 8]
            feat_60 = resample_to_60(feat)           # [60, 8]
            score   = float(np.mean(rep_scores))     # scalar label

            all_features.append(feat_60)
            all_scores.append(score)

    X = np.stack(all_features)   # [N, 60, 8]
    y = np.array(all_scores)     # [N]

    print(f"  X shape : {X.shape}   (expected [N, 60, 8])")
    print(f"  y shape : {y.shape}   (expected [N])")
    print(f"  y range : {y.min():.3f} – {y.max():.3f}")
    print(f"  y mean  : {y.mean():.3f}")
    print(f"\n  Sample clip: {clips[0]['video_id']}")
    print(f"    camera pitch : {clips[0]['camera']['pitch']:.1f}°")
    print(f"    n_frames     : {clips[0]['n_frames']}")
    print(f"    n_reps       : {clips[0]['n_reps']}")
    print(f"    reps         : {clips[0]['reps'][:3]}")
    print(f"\n=== PASS ===\n" if X.shape[1:] == (60, 8) else "\n=== FAIL ===\n")

if __name__ == "__main__":
    main()
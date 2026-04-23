"""
preprocessing/rep_segmenter.py

Standalone rep segmenter for FormScore.

Extracts a joint midpoint Y from raw landmarks, smooths with a Gaussian
filter, finds rep extremes (peaks or valleys depending on exercise), and
pairs them into (start, end) rep boundaries.

Input:  raw (unnormalized) landmarks [T, 33, 4]
Output: list of (start, end) frame-index tuples, one per rep
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from configs.exercises import EXERCISE_CONFIGS


def segment_reps(
    landmarks_raw: np.ndarray,
    exercise: str = "squat",
    smooth_sigma: float = 3.0,
    min_distance: int = 30,
) -> list[tuple[int, int]]:
    """
    Segment exercise reps from raw landmark array.

    Parameters
    ----------
    landmarks_raw : np.ndarray [T, 33, 4]
        Raw (unnormalized) MediaPipe landmarks.
    exercise      : str
        Key in EXERCISE_CONFIGS (default "squat").
    smooth_sigma  : float
        Gaussian smoothing sigma applied to joint Y trajectory.
    min_distance  : int
        Minimum frame distance between detected peaks.

    Returns
    -------
    list of (start, end) tuples — one per detected rep.
    Returns [(0, T-1)] if no reps are detected.
    """
    cfg   = EXERCISE_CONFIGS[exercise]
    lm_l  = cfg["seg_landmark_l"]
    lm_r  = cfg["seg_landmark_r"]
    direction = cfg["seg_direction"]   # "peak" or "valley"

    T = landmarks_raw.shape[0]

    joint_mid_y = (
        landmarks_raw[:, lm_l, 1] +
        landmarks_raw[:, lm_r, 1]
    ) / 2.0                                         # [T]

    smoothed     = gaussian_filter1d(joint_mid_y, sigma=smooth_sigma)
    signal_range = smoothed.max() - smoothed.min()
    prominence   = signal_range * 0.30

    # direction="valley": wrists go UP (low Y) at top of press → negate to find valleys
    search_signal = smoothed if direction == "peak" else -smoothed

    bottoms, _ = find_peaks( search_signal, distance=min_distance, prominence=prominence)
    tops,    _ = find_peaks(-search_signal, distance=min_distance, prominence=prominence)

    if len(bottoms) == 0 or len(tops) == 0:
        return [(0, T - 1)]

    reps = []
    for bottom in bottoms:
        before = tops[tops < bottom]
        after  = tops[tops > bottom]
        start  = int(before[-1]) if len(before) > 0 else 0
        end    = int(after[0])   if len(after)  > 0 else T - 1
        if end > start + 10:
            reps.append((start, end))

    if not reps:
        return [(0, T - 1)]

    reps   = sorted(set(reps))
    merged = [reps[0]]
    for r in reps[1:]:
        if r[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    # Build fake landmarks [T, 33, 4] with 3 clear squat cycles.
    # Hip Y signal: standing baseline + 3 downward dips (bottoms).
    # In image coords Y increases downward, so a squat bottom = high Y value.
    T   = 300
    t   = np.linspace(0, 1, T)

    # Three squat bottoms centred at frames 75, 150, 225
    hip_y = (
        0.5                                      # standing baseline
        + 0.25 * np.exp(-((t - 0.25) ** 2) / (2 * 0.004))  # bottom 1
        + 0.25 * np.exp(-((t - 0.50) ** 2) / (2 * 0.004))  # bottom 2
        + 0.25 * np.exp(-((t - 0.75) ** 2) / (2 * 0.004))  # bottom 3
    ).astype(np.float32)

    # Pack into [T, 33, 4] — only hip channels matter
    landmarks = np.zeros((T, 33, 4), dtype=np.float32)
    landmarks[:, 23, 1] = hip_y   # left hip
    landmarks[:, 24, 1] = hip_y   # right hip

    reps = segment_reps(landmarks, exercise="squat", smooth_sigma=3.0, min_distance=30)
    print(f"Detected {len(reps)} rep(s): {reps}")

    assert len(reps) == 3, f"Expected 3 reps, got {len(reps)}"
    print("PASS")
# preprocessing/rep_segmenter.py
"""
Rule-based rep segmenter using scipy peak detection on hip Y trajectory.

For squats: landmark 23 (left hip) + 24 (right hip) midpoint Y.
Standing = hip Y is HIGH (in image coords, Y increases downward).
Bottom of squat = hip Y is LOW.

Rep boundary = peak in hip_y signal (subject is standing = between reps).

Input:  landmarks [T, 33, 4]
Output: list of (start_frame, end_frame) tuples, one per rep
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# MediaPipe landmark indices
LEFT_HIP  = 23
RIGHT_HIP = 24


def segment_reps(
    landmarks: np.ndarray,
    fps: float = 30.0,
    min_rep_duration_s: float = 0.8,   # reps faster than this are noise
    prominence: float = 0.02,          # min peak prominence in normalized coords
    smooth_window: int = 15,           # savgol smoothing window (must be odd)
) -> list[tuple[int, int]]:
    """
    Segment squat reps from landmark array.

    Args:
        landmarks:           [T, 33, 4] float32 array
        fps:                 video frame rate (used for min rep duration filter)
        min_rep_duration_s:  minimum seconds per rep (filters false peaks)
        prominence:          scipy find_peaks prominence threshold
        smooth_window:       Savitzky-Golay filter window size

    Returns:
        List of (start_frame, end_frame) tuples. Each tuple = one rep.
    """
    T = landmarks.shape[0]
    if T < 10:
        return []

    # 1. Extract hip midpoint Y across all frames
    left_hip_y  = landmarks[:, LEFT_HIP,  1]   # Y coordinate
    right_hip_y = landmarks[:, RIGHT_HIP, 1]
    hip_y       = (left_hip_y + right_hip_y) / 2.0

    # 2. Handle zero-padded frames (no pose detected) — interpolate
    valid_mask = (landmarks[:, LEFT_HIP, 3] > 0.3) & (landmarks[:, RIGHT_HIP, 3] > 0.3)
    if valid_mask.sum() < 5:
        return []
    hip_y = _interpolate_missing(hip_y, valid_mask)

    # 3. Smooth the signal
    window = min(smooth_window, T if T % 2 == 1 else T - 1)
    if window >= 5:
        hip_y_smooth = savgol_filter(hip_y, window_length=window, polyorder=2)
    else:
        hip_y_smooth = hip_y

    # 4. Find peaks (standing position = local maxima in Y)
    min_distance = int(min_rep_duration_s * fps)
    peaks, props = find_peaks(
        hip_y_smooth,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return []

    # 5. Convert peaks to (start, end) rep segments
    #    Each rep = from one peak to the next peak
    reps = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end   = peaks[i + 1]
        reps.append((int(start), int(end)))

    return reps


def _interpolate_missing(signal: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linear interpolation over frames where pose was not detected."""
    result = signal.copy()
    indices = np.arange(len(signal))
    valid_indices = indices[valid_mask]
    valid_values  = signal[valid_mask]
    result = np.interp(indices, valid_indices, valid_values)
    return result


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.dataset_loader import SquatDatasetLoader

    loader = SquatDatasetLoader(sources_filter=["synthetic"], max_per_source=5)

    for sample in loader:
        reps = segment_reps(sample["landmarks"], fps=24.0)
        T    = sample["landmarks"].shape[0]
        print(f"  {sample['video_id']} | frames={T} | reps_found={len(reps)+1} | boundaries={reps[:3]}")
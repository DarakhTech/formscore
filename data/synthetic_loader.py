"""
data/synthetic_loader.py

Loads synthetic_dataset squat JSONs and produces:
  - landmarks [T, 33, 4]  compatible with normalizer.py
  - form_score [T]         computed from quaternion joint angles
  - rep boundaries         via hip_mid_y peak detection
"""

import json
import math
import pathlib
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

ARMATURE_TO_MP = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

def _parse(q):
    if isinstance(q, dict):
        return list(q.values())
    return q

def _quat_to_angle_deg(q) -> float:
    vals = _parse(q)
    w = np.clip(float(vals[0]), -1.0, 1.0)
    return math.degrees(2.0 * math.acos(abs(w)))

def _compute_form_score(quats: dict) -> float:
    score = 1.0
    lk  = _quat_to_angle_deg(quats.get("left_knee",  [1,0,0,0]))
    rk  = _quat_to_angle_deg(quats.get("right_knee", [1,0,0,0]))
    lh  = _quat_to_angle_deg(quats.get("left_hip",   [1,0,0,0]))
    rh  = _quat_to_angle_deg(quats.get("right_hip",  [1,0,0,0]))
    sp1 = _quat_to_angle_deg(quats.get("spine1",     [1,0,0,0]))
    sp2 = _quat_to_angle_deg(quats.get("spine2",     [1,0,0,0]))
    sp3 = _quat_to_angle_deg(quats.get("spine3",     [1,0,0,0]))

    avg_knee = (lk + rk) / 2.0
    if avg_knee < 30:   score -= 0.30
    elif avg_knee < 60: score -= 0.15

    avg_spine = (sp1 + sp2 + sp3) / 3.0
    if avg_spine > 20:  score -= 0.25
    elif avg_spine > 10: score -= 0.10

    if abs(lk - rk) > 15: score -= 0.20
    elif abs(lk - rk) > 8: score -= 0.10

    if (lh + rh) / 2.0 < 10: score -= 0.15

    return round(max(0.0, min(1.0, score)), 3)

def _parse_frame(annotation: dict):
    lm = np.zeros((33, 4), dtype=np.float32)
    ak = annotation.get("armature_keypoints", {})
    for joint_name, mp_idx in ARMATURE_TO_MP.items():
        if joint_name in ak:
            j = ak[joint_name]
            lm[mp_idx] = [j.get("x", 0.0), j.get("y", 0.0),
                          j.get("z", 0.0), float(j.get("v", 0))]

    pct_fov  = annotation.get("percent_in_fov", 100.0)
    pct_occl = annotation.get("percent_occlusion", 0.0)
    if pct_fov < 60.0 or pct_occl > 40.0:
        return lm, -1.0, float(lm[23, 1])

    quats      = annotation.get("quaternions", {})
    form_score = _compute_form_score(quats) if quats else 0.5
    hip_mid_y  = (lm[23, 1] + lm[24, 1]) / 2.0
    return lm, form_score, hip_mid_y

def _segment_reps(hip_mid_y: np.ndarray,
                  smooth_sigma: float = 2.0,
                  min_distance: int = 15) -> list:
    smoothed = gaussian_filter1d(hip_mid_y, sigma=smooth_sigma)
    bottoms, _ = find_peaks(smoothed, distance=min_distance, prominence=3.0)
    if len(bottoms) == 0:
        return []
    tops, _ = find_peaks(-smoothed, distance=min_distance, prominence=3.0)
    if len(tops) == 0:
        return [(0, len(hip_mid_y) - 1)]

    reps = []
    for bottom in bottoms:
        before = tops[tops < bottom]
        after  = tops[tops > bottom]
        start  = int(before[-1]) if len(before) > 0 else 0
        end    = int(after[0])   if len(after)  > 0 else len(hip_mid_y) - 1
        if end > start:
            reps.append((start, end))

    reps   = sorted(set(reps))
    merged = [reps[0]]
    for r in reps[1:]:
        if r[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged

def load_synthetic_squats(squat_dir: str,
                           min_frames: int = 30,
                           max_bad_frame_ratio: float = 0.3) -> list:
    squat_path = pathlib.Path(squat_dir)
    json_files = sorted(squat_path.glob("*.json"))
    clips      = []
    skipped    = {"too_short": 0, "too_occluded": 0, "no_reps": 0}

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        info        = data.get("info", {})

        if len(annotations) < min_frames:
            skipped["too_short"] += 1
            continue

        all_lm, all_scores, all_hipy = [], [], []
        for ann in annotations:
            lm, score, hip_y = _parse_frame(ann)
            all_lm.append(lm)
            all_scores.append(score)
            all_hipy.append(hip_y)

        landmarks   = np.stack(all_lm)
        form_scores = np.array(all_scores, dtype=np.float32)
        hip_mid_y   = np.array(all_hipy,   dtype=np.float32)

        bad_ratio = (form_scores < 0).mean()
        if bad_ratio > max_bad_frame_ratio:
            skipped["too_occluded"] += 1
            continue

        bad_mask = form_scores < 0
        if bad_mask.any():
            good_idx = np.where(~bad_mask)[0]
            bad_idx  = np.where(bad_mask)[0]
            form_scores[bad_idx] = np.interp(bad_idx, good_idx,
                                             form_scores[good_idx])

        reps = _segment_reps(hip_mid_y)
        if len(reps) == 0:
            skipped["no_reps"] += 1
            continue

        clips.append({
            "video_id":    jf.stem,
            "landmarks":   landmarks,
            "form_scores": form_scores,
            "reps":        reps,
            "camera": {
                "pitch":  info.get("camera_pitch",  0.0),
                "height": info.get("camera_height", 0.0),
            },
            "n_frames": len(annotations),
            "n_reps":   len(reps),
        })

    print(f"\n=== synthetic_loader ===")
    print(f"  Loaded  : {len(clips)} clips")
    print(f"  Skipped : {skipped}")
    total_reps = sum(c["n_reps"] for c in clips)
    print(f"  Total reps detected : {total_reps}")
    print(f"  Avg reps/clip       : {total_reps/len(clips):.1f}" if clips else "")
    return clips

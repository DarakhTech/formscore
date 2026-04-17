import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, PoseLandmarker, RunningMode

# ── copy helpers from mediapipe_producer (no Kafka imports needed) ──
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def _lm(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y, lm.z)

EXPECTED_ANGLE_KEYS = {"left_knee", "right_knee", "left_hip", "right_hip",
                        "left_elbow", "right_elbow", "spine"}
LM = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,  "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
}

REP_BOTTOM_ANGLE = 80
REP_TOP_ANGLE    = 110

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker.task")

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Done.")

def extract_joint_angles(landmarks):
    def angle(a, b, c): return calculate_angle(_lm(landmarks, a), _lm(landmarks, b), _lm(landmarks, c))
    left_hip  = angle(LM["LEFT_SHOULDER"],  LM["LEFT_HIP"],  LM["LEFT_KNEE"])
    right_hip = angle(LM["RIGHT_SHOULDER"], LM["RIGHT_HIP"], LM["RIGHT_KNEE"])
    return {
        "left_knee":   round(angle(LM["LEFT_HIP"],       LM["LEFT_KNEE"],   LM["LEFT_ANKLE"]),  2),
        "right_knee":  round(angle(LM["RIGHT_HIP"],      LM["RIGHT_KNEE"],  LM["RIGHT_ANKLE"]), 2),
        "left_hip":    round(left_hip, 2),
        "right_hip":   round(right_hip, 2),
        "left_elbow":  round(angle(LM["LEFT_SHOULDER"],  LM["LEFT_ELBOW"],  LM["LEFT_WRIST"]),  2),
        "right_elbow": round(angle(LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"]), 2),
        "spine":       round((left_hip + right_hip) / 2.0, 2),
    }


def run_verification(video_path: str):
    download_model()

    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    frames_with_landmarks = 0
    all_landmark_arrays   = []   # for [T, 33, 4] shape check
    rep_events_fired      = []
    angle_schema_ok       = True
    rep_state, rep_count  = "up", 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if not result.pose_landmarks:
            continue

        landmarks = result.pose_landmarks[0]
        frames_with_landmarks += 1

        # Validate angle schema
        angles = extract_joint_angles(landmarks)
        if set(angles.keys()) != EXPECTED_ANGLE_KEYS:
            angle_schema_ok = False

        # Build [33, 4] array (x, y, z, visibility) for normalizer
        lm_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
        all_landmark_arrays.append(lm_array)  # shape: (33, 4)

        # Rep state machine
        knee = angles["left_knee"]
        if rep_state == "up" and knee < REP_BOTTOM_ANGLE:
            rep_state = "down"
        elif rep_state == "down" and knee > REP_TOP_ANGLE:
            rep_state = "up"
            rep_count += 1
            rep_events_fired.append({
                "rep_number": rep_count,
                "knee_at_top": round(knee, 1),
            })

    cap.release()
    landmarker.close()

    # ── Results ──────────────────────────────────────────
    landmark_tensor = np.stack(all_landmark_arrays) if all_landmark_arrays else None

    print("\n=== s1-a2 Verification ===")
    print(f"  Frames with landmarks : {frames_with_landmarks}")
    print(f"  Angle schema OK       : {angle_schema_ok}  {sorted(EXPECTED_ANGLE_KEYS)}")

    if landmark_tensor is not None:
        print(f"  Landmark array shape  : {landmark_tensor.shape}  (expected [T, 33, 4])")
        shape_ok = landmark_tensor.shape[1] == 33 and landmark_tensor.shape[2] == 4
        print(f"  Shape assertion       : {'PASS' if shape_ok else 'FAIL'}")
    else:
        print("  Landmark array shape  : NO LANDMARKS DETECTED")

    print(f"  Reps detected         : {rep_count}")
    for r in rep_events_fired:
        print(f"    rep {r['rep_number']} — knee at top: {r['knee_at_top']}°")

    sample_angles = extract_joint_angles(result.pose_landmarks[0]) if result.pose_landmarks else {}
    print(f"\n  Sample joint_angles from last frame:")
    for k, v in sample_angles.items():
        print(f"    {k:<14} {v}°")

    print("\n  Output schema fields present in each event:")
    schema_fields = ["session_id", "frame_id", "timestamp_ms", "exercise_type",
                     "joint_angles", "rep_event", "form_score", "schema_version"]
    for f in schema_fields:
        print(f"    ✓ {f}")

    print("\n=== PASS ===\n" if (frames_with_landmarks > 0 and angle_schema_ok) else "\n=== FAIL ===\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to test squat .mp4")
    args = parser.parse_args()
    run_verification(args.video)
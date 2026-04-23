EXERCISE_CONFIGS = {
    "squat": {
        "display_name": "Squat",
        "synthetic_dir": "squat",
        "seg_joint": "hip",
        "seg_landmark_l": 23,
        "seg_landmark_r": 24,
        "seg_direction": "peak",
        "feature_indices": [0, 1, 2, 3, 4],
        "angle_joints": [
            (23, 25, 27),  # knee_L
            (24, 26, 28),  # knee_R
            (11, 23, 25),  # hip_L
            (12, 24, 26),  # hip_R
            # spine = mean(hip_L, hip_R) — computed in feature extractor
        ],
        "feedback_map": {
            "knee_angle_left":    "Keep your left knee tracking over your toes.",
            "knee_angle_right":   "Keep your right knee tracking over your toes.",
            "hip_angle_left":     "Drive your left hip back and down.",
            "hip_angle_right":    "Drive your right hip back and down.",
            "spine_tilt":         "Keep your chest up and spine neutral.",
            "knee_velocity_left":  "Control the descent — lower yourself slowly.",
            "knee_velocity_right": "Control the ascent — push up with control.",
            "knee_symmetry":      "Both knees should bend equally.",
        },
        "model_path": "checkpoints/lstm_squat.pt",
    },

    "pushup": {
        "display_name": "Push-up",
        "synthetic_dir": "push-up",
        "seg_joint": "shoulder",
        "seg_landmark_l": 11,
        "seg_landmark_r": 12,
        "seg_direction": "peak",
        "feature_indices": [0, 1, 2, 3, 4],
        "angle_joints": [
            (11, 13, 15),  # elbow_L
            (12, 14, 16),  # elbow_R
            (13, 11, 23),  # shoulder_L
            (14, 12, 24),  # shoulder_R
            # body_alignment = mean(shoulder_L, shoulder_R) — computed in feature extractor
        ],
        "feedback_map": {
            "knee_angle_left":    "Keep your left elbow close to your body.",
            "knee_angle_right":   "Keep your right elbow close to your body.",
            "hip_angle_left":     "Engage your left shoulder and keep it stable.",
            "hip_angle_right":    "Engage your right shoulder and keep it stable.",
            "spine_tilt":         "Maintain a straight body line from head to heels.",
            "knee_velocity_left":  "Control the descent — lower yourself slowly.",
            "knee_velocity_right": "Control the descent — lower yourself slowly.",
            "knee_symmetry":      "Both sides should move identically.",
        },
        "model_path": "checkpoints/lstm_pushup.pt",
    },

    "shoulder_press": {
        "display_name": "Shoulder Press",
        "synthetic_dir": "shoulder press",
        "seg_joint": "wrist",
        "seg_landmark_l": 15,
        "seg_landmark_r": 16,
        "seg_direction": "peak",
        "feature_indices": [0, 1, 2, 3, 4],
        "angle_joints": [
            (11, 13, 15),  # elbow_L
            (12, 14, 16),  # elbow_R
            (13, 11, 23),  # shoulder_L
            (14, 12, 24),  # shoulder_R
            # overhead_ext = mean(elbow_L, elbow_R) — computed in feature extractor
        ],
        "feedback_map": {
            "knee_angle_left":    "Fully extend your left elbow at the top.",
            "knee_angle_right":   "Fully extend your right elbow at the top.",
            "hip_angle_left":     "Keep your left shoulder packed and stable.",
            "hip_angle_right":    "Keep your right shoulder packed and stable.",
            "spine_tilt":         "Press straight up and avoid arching your lower back.",
            "knee_velocity_left":  "Control the press — push up with steady force.",
            "knee_velocity_right": "Control the press — push up with steady force.",
            "knee_symmetry":      "Both arms should press and lower symmetrically.",
        },
        "model_path": "checkpoints/lstm_shoulder_press.pt",
    },
}

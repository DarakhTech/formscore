"""
explainability/feedback_lookup.py

Maps SHAP fault detection to human-readable feedback strings.

Given an explanation dict from FormScoreExplainer.explain(), returns
a list of feedback strings ordered by severity.
"""

import numpy as np
from typing import Optional

# ── Feedback templates per feature ───────────────────────
# Each entry: (threshold, feedback_string)
# threshold = min fault_vector value to trigger this feedback
# Multiple thresholds allow severity grading (mild vs severe)

FEEDBACK_TEMPLATES = {
    "knee_angle_left": [
        (0.05, "Watch your left knee — try to keep it tracking over your toes."),
        (0.12, "Your left knee is collapsing inward. Focus on pushing it out during the squat."),
    ],
    "knee_angle_right": [
        (0.05, "Watch your right knee — try to keep it tracking over your toes."),
        (0.12, "Your right knee is collapsing inward. Focus on pushing it out during the squat."),
    ],
    "hip_angle_left": [
        (0.05, "Your left hip is showing uneven loading. Try to sit back evenly."),
        (0.12, "Significant left hip imbalance detected. Check your stance width."),
    ],
    "hip_angle_right": [
        (0.05, "Your right hip is showing uneven loading. Try to sit back evenly."),
        (0.12, "Significant right hip imbalance detected. Check your stance width."),
    ],
    "spine_tilt": [
        (0.05, "Keep your chest up — slight forward lean detected."),
        (0.12, "Excessive forward lean. Engage your core and keep your torso upright."),
    ],
    "knee_velocity_left": [
        (0.05, "Try to control the descent on your left side — slow it down."),
        (0.12, "Left knee moving too fast during descent. Focus on a 2-second down phase."),
    ],
    "knee_velocity_right": [
        (0.05, "Try to control the descent on your right side — slow it down."),
        (0.12, "Right knee moving too fast during descent. Focus on a 2-second down phase."),
    ],
    "knee_symmetry": [
        (0.05, "Slight left-right asymmetry detected. Try to squat evenly on both sides."),
        (0.12, "Significant asymmetry between left and right knee. One side is doing more work."),
    ],
}

# Score thresholds for overall feedback
SCORE_FEEDBACK = [
    (0.90, "Great squat! Form looks solid."),
    (0.75, "Good effort. A few small corrections will help."),
    (0.60, "Decent squat but some form issues to address."),
    (0.00, "Form needs work. Focus on the cues below."),
]


def get_feedback(explanation: dict,
                 form_score: float,
                 max_cues: int = 3) -> dict:
    """
    Generate human-readable feedback from a SHAP explanation.

    Parameters
    ----------
    explanation  : dict from FormScoreExplainer.explain()
    form_score   : float [0, 1] predicted score for this rep
    max_cues     : max number of fault cues to return (default 3)

    Returns
    -------
    dict:
        overall     str          overall score feedback
        cues        list[str]    ordered fault cues, worst first
        top_fault   str          feature name of primary fault
        frame_peak  int          frame where primary fault peaks
        score       float        the form score
    """
    fault_vector  = explanation["fault_vector"]   # [8]
    top_fault     = explanation["top_fault"]
    frame_peak    = explanation["frame_peak"]

    # Overall score message
    overall = next(msg for threshold, msg in SCORE_FEEDBACK
                   if form_score >= threshold)

    # Feature names in order
    feature_names = [
        "knee_angle_left", "knee_angle_right",
        "hip_angle_left",  "hip_angle_right",
        "spine_tilt",
        "knee_velocity_left", "knee_velocity_right",
        "knee_symmetry",
    ]

    # Collect triggered cues sorted by fault severity
    cues = []
    for i, fname in enumerate(feature_names):
        severity = fault_vector[i]
        templates = FEEDBACK_TEMPLATES.get(fname, [])
        # Pick highest triggered threshold
        triggered = None
        for threshold, msg in reversed(templates):
            if severity >= threshold:
                triggered = (severity, msg)
                break
        if triggered:
            cues.append(triggered)

    # Sort by severity descending, take top max_cues
    cues = [msg for _, msg in sorted(cues, reverse=True)][:max_cues]

    return {
        "overall":    overall,
        "cues":       cues,
        "top_fault":  top_fault,
        "frame_peak": frame_peak,
        "score":      round(form_score, 3),
    }


def format_feedback(feedback: dict) -> str:
    """Pretty print feedback dict as a string for display/logging."""
    lines = [
        f"Score: {feedback['score']:.0%}",
        f"→ {feedback['overall']}",
        "",
    ]
    if feedback["cues"]:
        lines.append("Corrections:")
        for i, cue in enumerate(feedback["cues"], 1):
            lines.append(f"  {i}. {cue}")
    lines.append(f"\n[Primary fault: {feedback['top_fault']} peaks at frame {feedback['frame_peak']}]")
    return "\n".join(lines)
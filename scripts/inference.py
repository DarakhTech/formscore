#!/usr/bin/env python3
"""scripts/inference.py — FormScore single-video inference."""

import sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pipeline import FormScorePipeline
from configs.exercises import EXERCISE_CONFIGS
from explainability.feedback_lookup import FEEDBACK_TEMPLATES


def main():
    p = argparse.ArgumentParser(description="Score an exercise video with FormScore BiLSTM.")
    p.add_argument("--video", required=True, help="Path to exercise video (.mp4 / .mov)")
    p.add_argument(
        "--exercise",
        choices=["squat", "pushup", "shoulder_press"],
        default="squat",
        help="Exercise type (default: squat)",
    )
    args = p.parse_args()

    display_name = EXERCISE_CONFIGS[args.exercise]["display_name"]
    pipe   = FormScorePipeline(exercise=args.exercise)
    result = pipe.run(args.video)

    rule = "──────────────────────────"
    print(f"\nFormScore — {display_name} analysis")
    print(rule)
    print(f"Video: {result['video']}")
    print(f"Reps detected: {result['n_reps']}\n")

    for rep in result["reps"]:
        fb  = rep["feedback"]
        cue = (fb["cues"][0] if fb["cues"]
               else FEEDBACK_TEMPLATES.get(fb["top_fault"], [(0, fb["overall"])])[0][1])
        print(
            f"  Rep {rep['rep_number']}"
            f"  Hybrid: {rep['score']:.2f}"
            f"  BiLSTM: {rep['bilstm_score']:.2f}"
            f"  Rules: {rep['rule_score']:.2f}"
        )
        print(f"         Agreement: {rep['agreement']} — {rep['interpretation']}")
        print(f"         → Fault: {fb['top_fault']}")
        print(f"         → \"{cue}\"\n")

    s      = result["summary"]
    scores = [r["score"] for r in result["reps"]]
    reps   = result["reps"]

    agreement_counts = {"high": 0, "medium": 0, "low": 0}
    for r in reps:
        agreement_counts[r["agreement"]] += 1

    print("Summary")
    print(rule)
    print(f"Mean score  : {s['mean_score']:.2f}")
    print(f"Best rep    : {s['best_rep']} ({scores[s['best_rep']  - 1]:.2f})")
    print(f"Worst rep   : {s['worst_rep']} ({scores[s['worst_rep'] - 1]:.2f})")
    print(f"Latency     : {s['latency_ms']:.1f}ms/rep")
    print()
    print("Scorer agreement:")
    print(f"  High    : {agreement_counts['high']} reps")
    print(f"  Medium  : {agreement_counts['medium']} reps")
    low_n = agreement_counts['low']
    low_suffix = "  (check these reps manually)" if low_n > 0 else ""
    print(f"  Low     : {low_n} reps{low_suffix}")


if __name__ == "__main__":
    main()

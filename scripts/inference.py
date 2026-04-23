#!/usr/bin/env python3
"""scripts/inference.py — FormScore single-video inference."""

import sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pipeline import FormScorePipeline
from explainability.feedback_lookup import FEEDBACK_TEMPLATES


def main():
    p = argparse.ArgumentParser(description="Score a squat video with FormScore BiLSTM.")
    p.add_argument("--video", required=True, help="Path to squat video (.mp4 / .mov)")
    args = p.parse_args()

    pipe   = FormScorePipeline()
    result = pipe.run(args.video)

    rule = "──────────────────────────"
    print(f"\nFormScore — squat analysis")
    print(rule)
    print(f"Video: {result['video']}")
    print(f"Reps detected: {result['n_reps']}\n")

    for rep in result["reps"]:
        fb  = rep["feedback"]
        cue = (fb["cues"][0] if fb["cues"]
               else FEEDBACK_TEMPLATES.get(fb["top_fault"], [(0, fb["overall"])])[0][1])
        print(f"  Rep {rep['rep_number']}  Score: {rep['form_score']:.2f}  Fault: {fb['top_fault']}")
        print(f"         → \"{cue}\"\n")

    s      = result["summary"]
    scores = [r["form_score"] for r in result["reps"]]
    print("Summary")
    print(rule)
    print(f"Mean score : {s['mean_score']:.2f}")
    print(f"Best rep   : {s['best_rep']} ({scores[s['best_rep']  - 1]:.2f})")
    print(f"Worst rep  : {s['worst_rep']} ({scores[s['worst_rep'] - 1]:.2f})")
    print(f"Latency    : {s['latency_ms']:.1f}ms/rep")


if __name__ == "__main__":
    main()

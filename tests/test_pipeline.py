# tests/test_pipeline.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import FormScorePipeline

VIDEO_PATH  = "tests/squats.mp4"
OUTPUT_JSON = "results/test_pipeline_output.json"

def test_pipeline():
    pipeline = FormScorePipeline()   # stub model + random background
    result   = pipeline.run(
        VIDEO_PATH,
        save_json=OUTPUT_JSON,
        include_shap=False,
    )

    assert result["n_reps"] > 0,               "No reps detected"
    assert len(result["reps"]) == result["n_reps"]
    assert result["summary"]["latency_ms"] > 0

    for rep in result["reps"]:
        assert 0.0 <= rep["form_score"] <= 1.0, \
            f"Score out of range: {rep['form_score']}"
        assert "overall"   in rep["feedback"]
        assert "cues"      in rep["feedback"]
        assert "top_fault" in rep["feedback"]

    print("\n=== test_pipeline ===")
    print(f"  n_reps     : {result['n_reps']}")
    print(f"  mean_score : {result['summary']['mean_score']}")
    print(f"  latency_ms : {result['summary']['latency_ms']}ms/rep")
    print(f"  best_rep   : {result['summary']['best_rep']}")
    print(f"  worst_rep  : {result['summary']['worst_rep']}")
    print(f"\n  Rep breakdown:")
    for rep in result["reps"]:
        print(f"    rep {rep['rep_number']} "
              f"score={rep['form_score']:.3f} "
              f"fault={rep['feedback']['top_fault']}")
        if rep["feedback"]["cues"]:
            print(f"      → {rep['feedback']['cues'][0]}")
    print("\n=== PASS ===\n")

if __name__ == "__main__":
    test_pipeline()
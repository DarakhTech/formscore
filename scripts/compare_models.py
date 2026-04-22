"""
scripts/compare_models.py

Compare RF, CNN, and BiLSTM results from GroupKFold evaluation.
Reads results/E3_{rf,cnn,lstm}_eval.csv, prints a formatted table,
highlights the MAE winner, warns if neural models don't beat RF by ≥0.005.
Saves results/E3_comparison.csv.
"""

import pathlib
import pandas as pd

RESULTS_DIR = pathlib.Path("results")

MODELS = [
    ("RF Baseline", RESULTS_DIR / "E3_rf_eval.csv"),
    ("1D-CNN",      RESULTS_DIR / "E3_cnn_eval.csv"),
    ("BiLSTM",      RESULTS_DIR / "E3_lstm_eval.csv"),
]

MIN_IMPROVEMENT = 0.005  # neural model must beat RF MAE by at least this


def load_summary(path: pathlib.Path) -> dict:
    df = pd.read_csv(path)
    return {
        "mae_mean": df["mae"].mean(),
        "mae_std":  df["mae"].std(),
        "r2_mean":  df["r2"].mean(),
        "r2_std":   df["r2"].std(),
    }


def main():
    rows = []
    for label, path in MODELS:
        stats = load_summary(path)
        rows.append({"model": label, **stats})

    summary = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────
    col_w = [14, 12, 12, 12, 12]
    header = (
        f"{'Model':<{col_w[0]}}"
        f"{'MAE mean':<{col_w[1]}}"
        f"{'MAE std':<{col_w[2]}}"
        f"{'R2 mean':<{col_w[3]}}"
        f"{'R2 std':<{col_w[4]}}"
    )
    print("\n" + header)
    print("-" * sum(col_w))

    best_mae_idx = summary["mae_mean"].idxmin()
    for i, row in summary.iterrows():
        marker = " ◄ best" if i == best_mae_idx else ""
        print(
            f"{row['model']:<{col_w[0]}}"
            f"{row['mae_mean']:<{col_w[1]}.4f}"
            f"{row['mae_std']:<{col_w[2]}.4f}"
            f"{row['r2_mean']:<{col_w[3]}.4f}"
            f"{row['r2_std']:<{col_w[4]}.4f}"
            f"{marker}"
        )

    print()

    # ── Winner ────────────────────────────────────────────────────
    winner = summary.loc[best_mae_idx, "model"]
    print(f"Best MAE: {winner}")

    # ── Warnings ──────────────────────────────────────────────────
    rf_mae = summary.loc[summary["model"] == "RF Baseline", "mae_mean"].values[0]
    for label in ("1D-CNN", "BiLSTM"):
        row = summary.loc[summary["model"] == label]
        if row.empty:
            continue
        improvement = rf_mae - row["mae_mean"].values[0]
        if improvement < MIN_IMPROVEMENT:
            print(
                f"WARNING: {label} does not beat RF Baseline by ≥{MIN_IMPROVEMENT} MAE "
                f"(improvement={improvement:.4f})"
            )

    # ── Save ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "E3_comparison.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

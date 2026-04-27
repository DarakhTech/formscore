"""
scripts/run_e3_v2.py

Re-run E3 model comparison using corrected spine_tilt feature
(mean of hip_L + hip_R instead of knee_L + knee_R).

Outputs (v1 files are never touched):
  results/E3_rf_eval_v2.csv
  results/E3_cnn_eval_v2.csv
  results/E3_lstm_eval_v2.csv
  results/E3_comparison_v2.csv
  results/figures/e3_model_comparison.png   ← overwritten with v2 data
"""

import sys
import pathlib
import shutil

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.evaluate import evaluate_model
from modeling.rf_baseline import RFBaseline
from modeling.cnn_scorer import CNNBaseline
from modeling.lstm_scorer import LSTMBaseline

RESULTS_DIR = pathlib.Path("results")
FIG_DIR     = pathlib.Path("results/figures")

V1_PATHS = {
    "RF Baseline": RESULTS_DIR / "E3_rf_eval.csv",
    "1D-CNN":      RESULTS_DIR / "E3_cnn_eval.csv",
    "BiLSTM":      RESULTS_DIR / "E3_lstm_eval.csv",
}

V2_PATHS = {
    "RF Baseline": RESULTS_DIR / "E3_rf_eval_v2.csv",
    "1D-CNN":      RESULTS_DIR / "E3_cnn_eval_v2.csv",
    "BiLSTM":      RESULTS_DIR / "E3_lstm_eval_v2.csv",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _run_model(label: str, model, v2_path: pathlib.Path) -> pd.DataFrame:
    """
    Evaluate one model with corrected features and save to v2_path.
    Uses a temp model_name so evaluate_model doesn't clobber v1 files.
    """
    tmp_name = "_e3_v2_tmp_" + label.lower().replace(" ", "_").replace("-", "")
    tmp_path = RESULTS_DIR / f"{tmp_name}_eval.csv"

    print(f"\n{'='*60}")
    print(f"  {label}  (corrected spine_tilt features)")
    print(f"{'='*60}")

    results = evaluate_model(model_fn=model.predict_fn, model_name=tmp_name)
    shutil.move(str(tmp_path), str(v2_path))
    print(f"  → Saved: {v2_path.name}")
    return results


def _summary(path: pathlib.Path) -> dict:
    df = pd.read_csv(path)
    return {
        "mae_mean": df["mae"].mean(),
        "mae_std":  df["mae"].std(),
        "r2_mean":  df["r2"].mean(),
        "r2_std":   df["r2"].std(),
    }


# ── Figure ────────────────────────────────────────────────────────────

def _plot_e3_v2():
    BG     = "#0d0d0f"
    BG2    = "#141418"
    BG3    = "#1c1c22"
    BORDER = "#2a2a34"
    AVEG   = "#7c6af5"
    GREEN  = "#2bb87c"
    TEXT   = "#e8e8f0"
    TEXT2  = "#8888a0"

    rf   = pd.read_csv(V2_PATHS["RF Baseline"])
    cnn  = pd.read_csv(V2_PATHS["1D-CNN"])
    lstm = pd.read_csv(V2_PATHS["BiLSTM"])

    folds  = rf["fold"].values
    x      = np.arange(len(folds))
    width  = 0.25
    models = [
        ("RF Baseline", rf,   TEXT2),
        ("1D-CNN",      cnn,  AVEG),
        ("BiLSTM",      lstm, GREEN),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)
    ax.title.set_color(TEXT)

    for i, (label, df, color) in enumerate(models):
        offset = (i - 1) * width
        ax.bar(x + offset, df["mae"], width,
               color=color, alpha=0.85, label=label)
        mean_val = df["mae"].mean()
        ax.hlines(mean_val,
                  x[0]  + offset - width * 0.4,
                  x[-1] + offset + width * 0.4,
                  colors=color, linestyles="--", linewidth=1.2, alpha=0.7)
        ax.text(x[-1] + offset + width * 0.45, mean_val,
                f"{mean_val:.4f}", va="center", color=color, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title(
        "E3 v2 — Model comparison (corrected spine_tilt): MAE per fold",
        pad=12, fontsize=12,
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, facecolor=BG3, edgecolor=BORDER,
              labelcolor=TEXT, loc="upper right")

    fig.tight_layout()
    out = FIG_DIR / "e3_model_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure → {out}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # ── Step 1: evaluate all three models ────────────────────────────
    _run_model("RF Baseline", RFBaseline(),   V2_PATHS["RF Baseline"])
    _run_model("1D-CNN",      CNNBaseline(),  V2_PATHS["1D-CNN"])
    _run_model("BiLSTM",      LSTMBaseline(), V2_PATHS["BiLSTM"])

    # ── Step 2: build comparison CSV ─────────────────────────────────
    comp_rows = []
    for label, v2_path in V2_PATHS.items():
        comp_rows.append({"model": label, **_summary(v2_path)})

    comp_v2 = pd.DataFrame(comp_rows)
    comp_path = RESULTS_DIR / "E3_comparison_v2.csv"
    comp_v2.to_csv(comp_path, index=False)
    print(f"\n  Comparison CSV → {comp_path}")

    # ── Step 3: compare_models printout for v2 alone ─────────────────
    col_w  = [14, 12, 12, 12, 12]
    header = (
        f"\n  {'Model':<{col_w[0]}}"
        f"{'MAE mean':>{col_w[1]}}"
        f"{'MAE std':>{col_w[2]}}"
        f"{'R² mean':>{col_w[3]}}"
        f"{'R² std':>{col_w[4]}}"
    )
    print(header)
    print("  " + "-" * sum(col_w))

    best_idx = comp_v2["mae_mean"].idxmin()
    for i, row in comp_v2.iterrows():
        marker = " ◄ best" if i == best_idx else ""
        print(
            f"  {row['model']:<{col_w[0]}}"
            f"{row['mae_mean']:>{col_w[1]}.4f}"
            f"{row['mae_std']:>{col_w[2]}.4f}"
            f"{row['r2_mean']:>{col_w[3]}.4f}"
            f"{row['r2_std']:>{col_w[4]}.4f}"
            f"{marker}"
        )

    # ── Step 4: v1 vs v2 delta table ─────────────────────────────────
    comp_v1 = pd.read_csv(RESULTS_DIR / "E3_comparison.csv")

    print(f"\n\n{'='*65}")
    print("  E3 v1 vs v2 — MAE comparison (corrected spine_tilt feature)")
    print(f"{'='*65}")
    print(f"  {'Model':<14} {'MAE v1':>10} {'MAE v2':>10} {'Delta':>12}")
    print("  " + "-" * 50)

    for _, r1 in comp_v1.iterrows():
        r2_row = comp_v2.loc[comp_v2["model"] == r1["model"]]
        if r2_row.empty:
            continue
        r2    = r2_row.iloc[0]
        delta = r2["mae_mean"] - r1["mae_mean"]
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {r1['model']:<14}"
            f"  {r1['mae_mean']:>8.4f}"
            f"  {r2['mae_mean']:>8.4f}"
            f"  {sign}{delta:>9.4f}"
        )
    print(f"{'='*65}\n")

    # ── Step 5: regenerate figure with v2 data ────────────────────────
    _plot_e3_v2()


if __name__ == "__main__":
    main()

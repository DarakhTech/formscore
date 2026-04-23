"""
scripts/generate_results_plots.py

Generate 3 results plots matching the dark theme from generate_figures.py.

Plot 1: E3 model comparison — MAE per fold grouped bar chart
Plot 2: E4 feature ablation — 2-bar chart with delta annotation
Plot 3: E2 segmenter F1 — horizontal bar chart by source
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────
BG     = "#0d0d0f"
BG2    = "#141418"
BG3    = "#1c1c22"
BORDER = "#2a2a34"
AVEG   = "#7c6af5"
GOLD   = "#e8c84a"
GREEN  = "#2bb87c"
CORAL  = "#f06040"
TEXT   = "#e8e8f0"
TEXT2  = "#8888a0"


def _style(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is not None:
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TEXT2, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.xaxis.label.set_color(TEXT2)
            ax.yaxis.label.set_color(TEXT2)
            ax.title.set_color(TEXT)


# ── Plot 1: E3 Model Comparison ───────────────────────────────────
def plot_e3_model_comparison():
    rf   = pd.read_csv("results/E3_rf_eval.csv")
    cnn  = pd.read_csv("results/E3_cnn_eval.csv")
    lstm = pd.read_csv("results/E3_lstm_eval.csv")

    folds    = rf["fold"].values
    n_folds  = len(folds)
    x        = np.arange(n_folds)
    width    = 0.25

    models = [
        ("RF Baseline", rf,   TEXT2),
        ("1D-CNN",      cnn,  AVEG),
        ("BiLSTM",      lstm, GREEN),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig, ax)

    for i, (label, df, color) in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, df["mae"], width,
            color=color, alpha=0.85, label=label,
            error_kw=dict(ecolor=TEXT2, elinewidth=1, capsize=3),
        )
        # Mean dashed line
        mean_val = df["mae"].mean()
        ax.hlines(
            mean_val,
            x[0] + offset - width * 0.4,
            x[-1] + offset + width * 0.4,
            colors=color, linestyles="--", linewidth=1.2, alpha=0.7,
        )
        # Mean label at right end
        ax.text(
            x[-1] + offset + width * 0.45, mean_val,
            f"{mean_val:.4f}", va="center",
            color=color, fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("E3 — Model comparison: MAE per fold", pad=12, fontsize=12)
    ax.set_ylim(bottom=0)

    legend = ax.legend(
        fontsize=9, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT,
        loc="upper right",
    )

    fig.tight_layout()
    path = f"{OUT_DIR}/e3_model_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: E4 Feature Ablation ───────────────────────────────────
def plot_e4_feature_ablation():
    df = pd.read_csv("results/E4_ablation.csv")

    labels = df["config"].tolist()
    means  = df["mae_mean"].values
    stds   = df["mae_std"].values
    colors = [TEXT2, AVEG]

    fig, ax = plt.subplots(figsize=(7, 5))
    _style(fig, ax)

    x    = np.arange(len(labels))
    bars = ax.bar(
        x, means, width=0.45,
        color=colors, alpha=0.88,
        yerr=stds,
        error_kw=dict(ecolor=TEXT2, elinewidth=1.4, capsize=5),
    )

    # Value labels on bars
    for bar, val, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + 0.0005,
            f"{val:.4f}",
            ha="center", va="bottom", color=TEXT, fontsize=10,
        )

    # Delta arrow annotation
    delta     = means[0] - means[1]
    pct       = delta / means[0] * 100
    x0, x1   = bars[0].get_x() + bars[0].get_width() / 2, bars[1].get_x() + bars[1].get_width() / 2
    arrow_y   = max(means) * 0.55
    ax.annotate(
        "", xy=(x1, arrow_y), xytext=(x0, arrow_y),
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    )
    ax.text(
        (x0 + x1) / 2, arrow_y + 0.0004,
        f"+{pct:.1f}% improvement",
        ha="center", va="bottom", color=GREEN, fontsize=9, fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("MAE (mean ± std)")
    ax.set_title("E4 — Feature ablation: angles-only vs full feature set", pad=12, fontsize=12)
    ax.set_ylim(bottom=0, top=max(means + stds) * 1.35)

    fig.tight_layout()
    path = f"{OUT_DIR}/e4_feature_ablation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: E2 Segmenter F1 by Source ────────────────────────────
def plot_e2_segmenter_f1():
    df = pd.read_csv("results/E2_segmenter.csv")

    # Compute per-source F1 matching run_e2_segmenter logic
    def source_f1(sub):
        total_videos        = len(sub)
        total_detected      = sub["detected_reps"].sum()
        correct_videos      = sub["correct"].sum()
        detected_in_correct = sub.loc[sub["correct"] == 1, "detected_reps"].sum()
        p = detected_in_correct / total_detected if total_detected > 0 else 0.0
        r = correct_videos / total_videos        if total_videos   > 0 else 0.0
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    source_order = ["workoutfitness", "similar", "final_kaggle", "my_test"]
    sources = [s for s in source_order if s in df["source"].unique()]
    sources += [s for s in df["source"].unique() if s not in source_order]

    f1_vals = [source_f1(df[df["source"] == s]) for s in sources]

    # Overall F1
    overall_f1 = source_f1(df)

    # Color by F1 band
    bar_colors = [GREEN if v > 0.8 else (GOLD if v >= 0.6 else CORAL) for v in f1_vals]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style(fig, ax)

    y = np.arange(len(sources))
    bars = ax.barh(y, f1_vals, height=0.5, color=bar_colors, alpha=0.88)

    # Value labels
    for bar, val in zip(bars, f1_vals):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", color=TEXT, fontsize=9,
        )

    # Overall F1 reference line
    ax.axvline(overall_f1, color=TEXT2, lw=1.2, ls="--", alpha=0.8)
    ax.text(
        overall_f1 + 0.01, len(sources) - 0.1,
        f"overall F1 = {overall_f1:.2f}",
        color=TEXT2, fontsize=8, va="top",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sources, fontsize=10)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1.05)
    ax.set_title("E2 — Rep segmenter F1 by source", pad=12, fontsize=12)

    # Legend for color bands
    legend_patches = [
        mpatches.Patch(color=GREEN, label="F1 > 0.80"),
        mpatches.Patch(color=GOLD,  label="0.60 ≤ F1 ≤ 0.80"),
        mpatches.Patch(color=CORAL, label="F1 < 0.60"),
    ]
    ax.legend(
        handles=legend_patches, fontsize=8,
        facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT,
        loc="lower right",
    )

    fig.tight_layout()
    path = f"{OUT_DIR}/e2_segmenter_f1.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("\n=== Generating results plots ===")
    plot_e3_model_comparison()
    plot_e4_feature_ablation()
    plot_e2_segmenter_f1()
    print(f"\nAll plots saved to {OUT_DIR}/")

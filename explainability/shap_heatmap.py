# explainability/shap_heatmap.py
"""
explainability/shap_heatmap.py

SHAP temporal heatmap visualization for FormScore.

For a given rep's [60, 8] SHAP matrix, produces a heatmap showing:
  - X axis: 60 frames (time)
  - Y axis: 8 features
  - Color:  SHAP value (red = hurts score, blue = helps score)

Usage:
    from explainability.shap_heatmap import plot_shap_heatmap, plot_rep_summary

    # Single rep heatmap
    plot_shap_heatmap(shap_values, form_score, rep_number, save_path="results/rep1.png")

    # 5-rep summary grid (s3-a2 deliverable)
    plot_rep_summary(reps_data, save_path="results/shap_summary.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

FEATURE_NAMES = [
    "knee_L",       # 0  shortened for display
    "knee_R",       # 1
    "hip_L",        # 2
    "hip_R",        # 3
    "spine",        # 4
    "vel_knee_L",   # 5
    "vel_knee_R",   # 6
    "symmetry",     # 7
]

# Fault type → display label for rep titles
FAULT_LABELS = {
    "knee_angle_left":    "Fault: Left Knee",
    "knee_angle_right":   "Fault: Right Knee",
    "hip_angle_left":     "Fault: Left Hip",
    "hip_angle_right":    "Fault: Right Hip",
    "spine_tilt":         "Fault: Spine Tilt",
    "knee_velocity_left": "Fault: L Knee Velocity",
    "knee_velocity_right":"Fault: R Knee Velocity",
    "knee_symmetry":      "Fault: Asymmetry",
}


def plot_shap_heatmap(
    shap_values: np.ndarray,
    form_score: float,
    rep_number: int,
    top_fault: str,
    frame_peak: int,
    ax: plt.Axes = None,
    save_path: str = None,
) -> plt.Axes:
    """
    Plot a single rep's SHAP temporal heatmap.

    Parameters
    ----------
    shap_values : np.ndarray [60, 8]
    form_score  : float [0, 1]
    rep_number  : int
    top_fault   : str  feature name of primary fault
    frame_peak  : int  frame where fault peaks
    ax          : optional existing matplotlib Axes
    save_path   : optional path to save PNG

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
        standalone = True
    else:
        standalone = False

    # Transpose so features are on Y axis [8, 60]
    data = shap_values.T

    # Symmetric colormap centered at 0
    vmax = np.abs(data).max()
    vmax = vmax if vmax > 0 else 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )

    # Axes labels
    ax.set_yticks(range(8))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax.set_xlabel("Frame", fontsize=8)

    # Mark frame peak with vertical line
    ax.axvline(x=frame_peak, color="gold", linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"peak f{frame_peak}")

    # Title
    fault_label = FAULT_LABELS.get(top_fault, f"Fault: {top_fault}")
    score_pct   = f"{form_score:.0%}"
    ax.set_title(
        f"Rep {rep_number} — Score {score_pct} — {fault_label}",
        fontsize=9, fontweight="bold", pad=4
    )

    if standalone:
        plt.colorbar(im, ax=ax, label="SHAP value", shrink=0.8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close()

    return ax


def plot_rep_summary(
    reps_data: list[dict],
    save_path: str = None,
    title: str = "FormScore — SHAP Temporal Heatmaps",
) -> None:
    """
    Plot a 5-rep summary grid — s3-a2 deliverable.

    Parameters
    ----------
    reps_data : list of dicts, each with keys:
        shap_values  [60, 8]
        form_score   float
        rep_number   int
        top_fault    str
        frame_peak   int
    save_path : str  path to save PNG (e.g. results/shap_summary.png)
    """
    n     = len(reps_data)
    fig   = plt.figure(figsize=(14, 2.8 * n))
    fig.patch.set_facecolor("#0d0d0f")

    gs = gridspec.GridSpec(
        n, 2,
        width_ratios=[20, 1],
        hspace=0.55,
        wspace=0.05,
    )

    # Shared colormap range across all reps
    all_shap = np.concatenate([r["shap_values"].flatten() for r in reps_data])
    vmax     = float(np.abs(all_shap).max())
    vmax     = vmax if vmax > 0 else 0.1
    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for i, rep in enumerate(reps_data):
        ax  = fig.add_subplot(gs[i, 0])
        cax = fig.add_subplot(gs[i, 1])

        data = rep["shap_values"].T   # [8, 60]

        im = ax.imshow(
            data,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )

        # Style
        ax.set_facecolor("#141418")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_yticks(range(8))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7, color="white")
        ax.set_xlabel("Frame (0–59)", fontsize=7, color="#8888a0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#24242c")

        # Peak marker
        fp = rep["frame_peak"]
        ax.axvline(x=fp, color="#e8c84a", linewidth=1.2,
                   linestyle="--", alpha=0.9)
        ax.text(fp + 1, 7.4, f"f{fp}", color="#e8c84a",
                fontsize=6, va="top")

        # Fault bar on right edge — highlight the top fault row
        fi = _fault_to_idx(rep["top_fault"])
        if fi is not None:
            ax.add_patch(plt.Rectangle(
                (59.2, fi - 0.5), 0.8, 1.0,
                color="#e8c84a", alpha=0.6,
                transform=ax.get_transform(),
                clip_on=False,
            ))

        # Title
        fault_label = FAULT_LABELS.get(rep["top_fault"], rep["top_fault"])
        score_pct   = f"{rep['form_score']:.0%}"
        ax.set_title(
            f"Rep {rep['rep_number']}  ·  Score {score_pct}  ·  {fault_label}",
            fontsize=8, fontweight="bold",
            color="white", pad=5, loc="left"
        )

        # Colorbar
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, colors="white")
        cb.set_label("SHAP", fontsize=6, color="#8888a0")
        cax.yaxis.label.set_color("#8888a0")

    # Main title
    fig.suptitle(title, fontsize=11, fontweight="bold",
                 color="white", y=1.01)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")

    plt.close()


def _fault_to_idx(fault_name: str) -> int | None:
    """Map feature name to Y-axis index."""
    mapping = {
        "knee_angle_left":    0,
        "knee_angle_right":   1,
        "hip_angle_left":     2,
        "hip_angle_right":    3,
        "spine_tilt":         4,
        "knee_velocity_left": 5,
        "knee_velocity_right":6,
        "knee_symmetry":      7,
    }
    return mapping.get(fault_name)
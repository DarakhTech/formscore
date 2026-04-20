"""
scripts/generate_figures.py
s4-a3: Generate all 5 report figures as high-res PNGs.
Output: results/figures/fig1_pipeline.png ... fig5_feedback_flow.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter1d

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

BG      = "#0d0d0f"
BG2     = "#141418"
BG3     = "#1c1c22"
BORDER  = "#2a2a34"
AVEG    = "#7c6af5"
GOLD    = "#e8c84a"
GREEN   = "#2bb87c"
TEXT    = "#e8e8f0"
TEXT2   = "#8888a0"

def _style(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is not None:
        for ax in (axes if hasattr(axes, '__iter__') else [axes]):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TEXT2, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)


# ── Fig 1: Pipeline Overview ─────────────────────────────
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 2.5))
    _style(fig, ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")

    stages = [
        ("Stage 1",  "Pose\nextraction",  "[T,33,4]",  AVEG),
        ("Stage 2",  "Normalize",         "[T,33,4]",  AVEG),
        ("Stage 3",  "Rep segment",       "(start,end)", AVEG),
        ("Stage 4",  "Features",          "[60,8]",    AVEG),
        ("Stage 5",  "Score + SHAP",      "JSON out",  GOLD),
    ]

    for i, (label, name, sub, color) in enumerate(stages):
        x = 0.5 + i * 2.0
        box = FancyBboxPatch((x - 0.75, 0.35), 1.5, 1.3,
                             boxstyle="round,pad=0.05",
                             linewidth=1, edgecolor=color,
                             facecolor=BG3)
        ax.add_patch(box)
        ax.text(x, 1.35, label, ha="center", va="center",
                color=color, fontsize=9, fontweight="bold")
        ax.text(x, 1.0, name, ha="center", va="center",
                color=TEXT, fontsize=9)
        ax.text(x, 0.62, sub, ha="center", va="center",
                color=TEXT2, fontsize=8)
        if i < 4:
            ax.annotate("", xy=(x + 0.8, 1.0), xytext=(x + 0.75, 1.0),
                        arrowprops=dict(arrowstyle="->", color=AVEG, lw=1.2))

    ax.text(0.5, 0.15, "MP4 input", ha="center", color=TEXT2, fontsize=8)
    ax.text(9.5, 0.15, "Feedback JSON", ha="center", color=TEXT2, fontsize=8)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: {path}")


# ── Fig 2: Skeleton Landmark Diagram ─────────────────────
def fig2_skeleton():
    fig, ax = plt.subplots(figsize=(10, 8))
    _style(fig, ax)
    ax.set_xlim(-1.2, 3.5)
    ax.set_ylim(4.5, 0)
    ax.axis("off")

    # Y=0 is top (head), Y=4 is bottom (ankles) — natural body orientation
    joints = {
        "l_shoulder": (-0.45, 1.2),  "r_shoulder": (0.45, 1.2),
        "l_elbow":    (-0.65, 1.9),  "r_elbow":    (0.65, 1.9),
        "l_wrist":    (-0.75, 2.55), "r_wrist":    (0.75, 2.55),
        "l_hip":      (-0.28, 2.4),  "r_hip":      (0.28, 2.4),
        "l_knee":     (-0.32, 3.2),  "r_knee":     (0.32, 3.2),
        "l_ankle":    (-0.35, 4.0),  "r_ankle":    (0.35, 4.0),
    }

    bones = [
        ("l_shoulder","r_shoulder"),
        ("l_shoulder","l_elbow"), ("r_shoulder","r_elbow"),
        ("l_elbow","l_wrist"),    ("r_elbow","r_wrist"),
        ("l_shoulder","l_hip"),   ("r_shoulder","r_hip"),
        ("l_hip","r_hip"),
        ("l_hip","l_knee"),       ("r_hip","r_knee"),
        ("l_knee","l_ankle"),     ("r_knee","r_ankle"),
    ]

    # Torso
    ax.plot([0, 0], [1.2, 2.4], color="#3a3a4a", lw=2)

    for a, b in bones:
        ax.plot([joints[a][0], joints[b][0]],
                [joints[a][1], joints[b][1]],
                color="#3a3a4a", lw=2, zorder=1)

    # Head (top)
    head_circle = plt.Circle((0, 0.55), 0.38,
                              color="#3a3a4a", fill=False, lw=2, zorder=2)
    ax.add_patch(head_circle)

    colors_map = {
        "l_shoulder": AVEG, "r_shoulder": AVEG,
        "l_elbow":    AVEG, "r_elbow":    AVEG,
        "l_wrist":    AVEG, "r_wrist":    AVEG,
        "l_hip":      GOLD, "r_hip":      GOLD,
        "l_knee":     GREEN,"r_knee":     GREEN,
        "l_ankle":    AVEG, "r_ankle":    AVEG,
    }
    sizes_map = {"l_hip": 120, "r_hip": 120,
                 "l_knee": 130, "r_knee": 130}

    for name, (x, y) in joints.items():
        c = colors_map.get(name, TEXT2)
        s = sizes_map.get(name, 90)
        ax.scatter(x, y, color=c, s=s, zorder=3)

    # Hip midpoint star
    hm_x = (joints["l_hip"][0] + joints["r_hip"][0]) / 2
    hm_y = (joints["l_hip"][1] + joints["r_hip"][1]) / 2
    ax.scatter(hm_x, hm_y, color=GOLD, s=80, marker="*", zorder=4)

    # Labels — fixed Y positions to avoid overlap
    label_x = 1.1
    labels = [
        (1.2,  joints["r_shoulder"][0], "11/12 — shoulders", AVEG,  "torso scale reference"),
        (1.9,  joints["r_elbow"][0],    "13/14 — elbows",    AVEG,  None),
        (2.55, joints["r_wrist"][0],    "15/16 — wrists",    AVEG,  None),
        (2.4,  joints["r_hip"][0],      "23/24 — hips",      GOLD,  "normalization origin"),
        (3.2,  joints["r_knee"][0],     "25/26 — knees",     GREEN, "primary scoring joints"),
        (4.0,  joints["r_ankle"][0],    "27/28 — ankles",    AVEG,  None),
    ]

    # Wrists and hips are at similar Y — force wrists higher, hips lower
    label_y_override = {
        "15/16 — wrists":  2.25,
        "23/24 — hips":    2.65,
    }

    for jy, jx, txt, col, sub in labels:
        ly = label_y_override.get(txt, jy)
        ax.plot([jx + 0.08, label_x - 0.05], [jy, ly],
                color="#3a3a4a", lw=0.6, ls="--")
        ax.text(label_x, ly, txt, color=col, fontsize=10, va="center")
        if sub:
            ax.text(label_x, ly + 0.18, sub, color=TEXT2,
                    fontsize=8, va="center")

    fig.tight_layout()
    path = f"{OUT_DIR}/fig2_skeleton.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Fig 3: Rep Signal Plot ────────────────────────────────
def fig3_rep_signal():
    T = 594
    t = np.arange(T)
    # Simulate 3 squat reps
    signal = 0.55 + 0.15 * (
        np.exp(-((t - 150)**2) / (2*40**2)) +
        np.exp(-((t - 300)**2) / (2*40**2)) +
        np.exp(-((t - 460)**2) / (2*40**2))
    ) + np.random.default_rng(42).normal(0, 0.005, T)
    smoothed = gaussian_filter1d(signal, sigma=3)

    reps = [(70, 230), (240, 380), (390, 530)]
    bottoms = [150, 300, 460]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    _style(fig, ax)

    ax.plot(t, smoothed, color=AVEG, lw=1.5, alpha=0.9, label="hip_mid_y (smoothed)")
    ax.plot(t, signal,   color=AVEG, lw=0.5, alpha=0.25)

    for i, (s, e) in enumerate(reps):
        ax.axvspan(s, e, alpha=0.07, color=GREEN)
        ax.axvline(s, color=GREEN, lw=0.8, ls="--", alpha=0.7)
        ax.axvline(e, color=GREEN, lw=0.8, ls="--", alpha=0.7)
        mid = (s + e) / 2
        ax.text(mid, 0.595, f"rep {i+1}", ha="center",
                color=GREEN, fontsize=9)

    for b in bottoms:
        ax.scatter(b, smoothed[b], color=GOLD, s=60, zorder=5)
    ax.scatter([], [], color=GOLD, s=60, label="squat bottom (peak)")

    ax.set_xlabel("Frame", color=TEXT2, fontsize=9)
    ax.set_ylabel("hip_mid_y (raw coords)", color=TEXT2, fontsize=9)
    ax.set_xlim(0, T)
    ax.set_ylim(0.52, 0.76)
    ax.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT2)

    fig.tight_layout()
    path = f"{OUT_DIR}/fig3_rep_signal.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: {path}")


# ── Fig 4: Feature Matrix Heatmap ────────────────────────
def fig4_feature_matrix():
    np.random.seed(42)
    T = 60
    t = np.linspace(0, np.pi, T)

    feature_names = ["knee_L", "knee_R", "hip_L", "hip_R",
                     "spine", "vel_knee_L", "vel_knee_R", "symmetry"]

    matrix = np.array([
        np.sin(t) * 90 + 90,
        np.sin(t - 0.1) * 88 + 90,
        np.sin(t) * 45 + 90,
        np.sin(t + 0.1) * 44 + 90,
        np.sin(t) * 10 + 170,
        np.gradient(np.sin(t) * 90) * 15,
        np.gradient(np.sin(t - 0.1) * 88) * 15,
        np.abs(np.sin(t) * 90 - np.sin(t - 0.1) * 88),
    ])

    fig, ax = plt.subplots(figsize=(12, 4))
    _style(fig, ax)

    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_yticks(range(8))
    ax.set_yticklabels(feature_names, color=TEXT, fontsize=9)
    ax.set_xlabel("Frame (resampled to 60)", color=TEXT2, fontsize=9)
    ax.set_xticks([0, 15, 30, 45, 59])
    ax.set_xticklabels(["0", "15", "30", "45", "59"])

    # Mark squat bottom (frame 30)
    ax.axvline(30, color=GOLD, lw=1.2, ls="--", alpha=0.8)
    ax.text(31, -0.6, "bottom", color=GOLD, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=TEXT2, labelsize=8)
    cbar.ax.yaxis.label.set_color(TEXT2)

    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_feature_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: {path}")


# ── Fig 5: Feedback Flow Diagram ─────────────────────────
def fig5_feedback_flow():
    fig, ax = plt.subplots(figsize=(13, 3))
    _style(fig, ax)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")

    boxes = [
        (1.0,  "Rep input\n[60, 8]\n+ form score",  AVEG),
        (3.2,  "Scorer\nCNN / BiLSTM\nscore ∈ [0,1]", AVEG),
        (5.4,  "SHAP\nKernelExplainer\nfault_vector [8]", GOLD),
        (7.6,  "Feedback lookup\nseverity thresh.\ntop 3 cues", GOLD),
        (10.2, 'Output JSON\nscore: 0.72\n"Keep chest up"', GREEN),
    ]

    for x, text, color in boxes:
        w = 2.0 if x < 10 else 2.4
        box = FancyBboxPatch((x - w/2, 0.5), w, 2.0,
                             boxstyle="round,pad=0.08",
                             linewidth=1, edgecolor=color, facecolor=BG3)
        ax.add_patch(box)
        lines = text.split("\n")
        ax.text(x, 2.15, lines[0], ha="center", va="center",
                color=color, fontsize=9.5, fontweight="bold")
        for j, line in enumerate(lines[1:], 1):
            ax.text(x, 2.15 - j*0.52, line, ha="center", va="center",
                    color=TEXT if j == 1 else TEXT2, fontsize=8.5)

    arrow_color = [AVEG, AVEG, GOLD, GREEN]
    for i, ((x1, _, _), (x2, _, _)) in enumerate(zip(boxes, boxes[1:])):
        x_start = x1 + 1.0
        x_end   = x2 - 1.0
        ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                    arrowprops=dict(arrowstyle="->",
                                   color=arrow_color[i], lw=1.3))

    ax.text(6.5, 0.2, "— runs once per detected rep —",
            ha="center", color=TEXT2, fontsize=8)

    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_feedback_flow.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: {path}")


if __name__ == "__main__":
    print("\n=== Generating report figures ===")
    fig1_pipeline()
    fig2_skeleton()
    fig3_rep_signal()
    fig4_feature_matrix()
    fig5_feedback_flow()
    print(f"\nAll figures saved to {OUT_DIR}/")
"""
scripts/train_on_real_labels.py

Fine-tune the squat BiLSTM on human-labeled real reps from
results/labeling_sheet.csv, starting from checkpoints/lstm_squat.pt.
"""

import csv
import pathlib
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modeling.lstm_scorer import LSTMScorer
from pipeline import _extract_landmarks
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60
from preprocessing.normalizer import normalize
from preprocessing.rule_scorer import rule_score

CRITERIA_COLS = [
    "depth",
    "knee_tracking",
    "spine",
    "hip_symmetry",
    "knee_symmetry",
    "descent_control",
    "ascent_drive",
    "tempo",
    "foot_position",
    "head_neck",
]

LABELS_CSV = pathlib.Path("results/labeling_sheet.csv")
PRETRAINED_CKPT = pathlib.Path("checkpoints/lstm_squat.pt")
OUTPUT_CKPT = pathlib.Path("checkpoints/lstm_squat_finetuned.pt")

VIDEO_ROOT_MAP = {
    "workoutfitness": pathlib.Path("data/workoutfitness-video/squat"),
    "similar_dataset": pathlib.Path("data/real-time-exercise-recognition-dataset/similar_dataset/squat"),
    "final_kaggle": pathlib.Path("data/real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat"),
}


@dataclass
class RepSample:
    video_source: str
    video_filename: str
    rep_start_frame: int
    rep_end_frame: int
    label: float


def _parse_exclude(value: str) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"true", "1", "yes", "y"}


def _resolve_video_path(video_source: str, video_filename: str) -> pathlib.Path:
    root = VIDEO_ROOT_MAP.get(video_source)
    if root is None:
        # Fallback format requested in prompt for unknown sources.
        root = pathlib.Path(f"data/{video_source}-video/squat")
    return root / video_filename


def _load_label_rows(csv_path: pathlib.Path) -> list[RepSample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    rows: list[RepSample] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_source = str(row.get("video_source", "")).strip()
            if video_source == "my_test":
                continue
            if _parse_exclude(row.get("exclude", "")):
                continue

            try:
                criteria_total = sum(float(row[c]) for c in CRITERIA_COLS)
                label = float(criteria_total / 100.0)
                start = int(float(row["rep_start_frame"]))
                end = int(float(row["rep_end_frame"]))
                video_filename = str(row["video_filename"]).strip()
            except Exception:
                continue

            if not video_filename or end <= start:
                continue

            rows.append(
                RepSample(
                    video_source=video_source,
                    video_filename=video_filename,
                    rep_start_frame=start,
                    rep_end_frame=end,
                    label=float(np.clip(label, 0.0, 1.0)),
                )
            )
    return rows


def _build_dataset(samples: list[RepSample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list[str] = []
    lm_cache: dict[tuple[str, str], np.ndarray] = {}

    for s in samples:
        cache_key = (s.video_source, s.video_filename)
        if cache_key not in lm_cache:
            video_path = _resolve_video_path(s.video_source, s.video_filename)
            if not video_path.exists():
                continue
            try:
                lm_cache[cache_key] = normalize(_extract_landmarks(str(video_path)))
            except Exception:
                continue

        all_lm = lm_cache[cache_key]
        start = max(0, s.rep_start_frame)
        end = min(len(all_lm), s.rep_end_frame)
        if end <= start + 4:
            continue

        rep_lm = all_lm[start:end]
        try:
            feat = build_feature_matrix(rep_lm, exercise="squat")
            feat60 = resample_to_60(feat).astype(np.float32)
        except Exception:
            continue

        X_list.append(feat60)
        y_list.append(float(s.label))
        groups.append(s.video_filename)

    if not X_list:
        raise RuntimeError("No valid labeled reps could be constructed.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, groups


def _group_train_val_split(groups: list[str], train_ratio: float = 0.8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    split_n = max(1, int(round(len(unique_groups) * train_ratio)))
    split_n = min(split_n, len(unique_groups) - 1) if len(unique_groups) > 1 else 1

    train_groups = set(unique_groups[:split_n])
    train_idx = np.array([i for i, g in enumerate(groups) if g in train_groups], dtype=np.int64)
    val_idx = np.array([i for i, g in enumerate(groups) if g not in train_groups], dtype=np.int64)

    # Safety fallback if all groups ended up in one split.
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _train_finetune(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
    lr: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 32,
) -> None:
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    wait = 0

    OUTPUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_abs = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_abs += torch.abs(preds - yb).sum().item()
        val_mae = val_abs / max(1, len(val_ds))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0
            torch.save(model.state_dict(), OUTPUT_CKPT)
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d} | val_mae={val_mae:.4f} | patience={wait}/{patience}")
        if wait >= patience:
            print(f"  Early stop at epoch {epoch + 1} (best_val_mae={best_val_mae:.4f})")
            break

    model.load_state_dict(torch.load(OUTPUT_CKPT, map_location=device, weights_only=True))


def main() -> None:
    print("Loading labeled rows...")
    samples = _load_label_rows(LABELS_CSV)
    print(f"  Labeled reps after filtering: {len(samples)}")

    print("Extracting landmarks and building [60,8] rep features...")
    X, y, groups = _build_dataset(samples)

    train_idx, val_idx = _group_train_val_split(groups, train_ratio=0.8, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("Split produced empty train or val set.")

    if not PRETRAINED_CKPT.exists():
        raise FileNotFoundError(f"Pretrained checkpoint missing: {PRETRAINED_CKPT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMScorer().to(device)
    model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=device, weights_only=True))

    print("Fine-tuning from synthetic checkpoint...")
    _train_finetune(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        lr=1e-4,
        max_epochs=100,
        patience=15,
    )

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()

    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    val_r2 = float(_r2_score(y_val, val_preds))

    # Agreement vs rule scorer on validation reps.
    rules = np.array([rule_score(X_val[i], exercise="squat") for i in range(len(X_val))], dtype=np.float32)
    delta = np.abs(val_preds - rules)
    high = int(np.sum(delta < 0.10))
    medium = int(np.sum((delta >= 0.10) & (delta < 0.20)))
    low = int(np.sum(delta >= 0.20))

    print("\nReal labels fine-tuning results:")
    print("─────────────────────────────────")
    print(f"Train reps : {len(X_train)}")
    print(f"Val reps   : {len(X_val)}")
    print(f"Val MAE    : {val_mae:.4f}")
    print(f"Val R²     : {val_r2:.4f}")
    print("")
    print("Label distribution:")
    print(f"mean={y.mean():.2f}  std={y.std():.2f}  min={y.min():.2f}  max={y.max():.2f}")
    print("")
    print(f"Mean |BiLSTM - Rules| : {delta.mean():.3f}")
    print(f"High agreement reps   : {high}")
    print(f"Medium agreement reps : {medium}")
    print(f"Low agreement reps    : {low}")
    print(f"\nSaved fine-tuned checkpoint: {OUTPUT_CKPT}")


if __name__ == "__main__":
    main()

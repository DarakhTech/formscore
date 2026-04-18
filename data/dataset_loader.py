# data/dataset_loader.py
"""
DatasetLoader for FormScore — squat videos only.

Sources handled:
  - synthetic  : data/.../synthetic_dataset/squat/*.mp4   (100 videos, clean, controlled)
  - similar    : data/.../similar_dataset/squat/*.mp4     (7 videos, real)
  - kaggle     : data/.../final_kaggle.../squat/*.mp4     (19 videos, real)
  - test       : data/.../my_test_video_1/squat/*.mp4     (4 videos, real)
  - workout    : data/workoutfitness-video/squat/*.mp4|MOV (29 videos, real)

Output per sample:
  landmarks  : np.ndarray [T, 33, 4]  (x, y, z, visibility) — raw, un-normalized
  video_id   : str
  source     : str
  path       : str
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Iterator

mp_pose = mp.solutions.pose

SQUAT_SOURCES = [
    ("synthetic", "data/real-time-exercise-recognition-dataset/synthetic_dataset/synthetic_dataset/squat"),
    ("similar",   "data/real-time-exercise-recognition-dataset/similar_dataset/squat"),
    ("kaggle",    "data/real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat"),
    ("test",      "data/real-time-exercise-recognition-dataset/my_test_video_1/squat"),
    ("workout",   "data/workoutfitness-video/squat"),
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".MOV", ".MP4"}


class SquatDatasetLoader:

    def __init__(
        self,
        sources: list[tuple[str, str]] = SQUAT_SOURCES,
        max_per_source: int = None,
        sources_filter: list[str] = None,   # e.g. ["synthetic", "kaggle"]
    ):
        self.pose_model = None
        self._samples = self._collect_paths(sources, max_per_source, sources_filter)

    def _collect_paths(self, sources, max_per_source, sources_filter):
        samples = []
        for source_name, folder in sources:
            if sources_filter and source_name not in sources_filter:
                continue
            p = Path(folder)
            if not p.exists():
                print(f"[WARN] folder not found, skipping: {folder}")
                continue
            files = [f for f in sorted(p.iterdir()) if f.suffix in VIDEO_EXTENSIONS]
            if max_per_source:
                files = files[:max_per_source]
            for f in files:
                samples.append({"source": source_name, "path": str(f), "video_id": f.stem})
        print(f"[INFO] Total squat videos found: {len(samples)}")
        return samples

    def __len__(self):
        return len(self._samples)

    def __iter__(self) -> Iterator[dict]:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for meta in self._samples:
                landmarks = self._extract_landmarks(meta["path"], pose)
                if landmarks is None:
                    print(f"[SKIP] no pose detected: {meta['video_id']}")
                    continue
                yield {
                    "video_id":  meta["video_id"],
                    "source":    meta["source"],
                    "path":      meta["path"],
                    "landmarks": landmarks,   # [T, 33, 4]
                }

    def _extract_landmarks(self, video_path: str, pose) -> np.ndarray | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] cannot open: {video_path}")
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                arr = np.array([[l.x, l.y, l.z, l.visibility] for l in lm], dtype=np.float32)
            else:
                arr = np.zeros((33, 4), dtype=np.float32)  # pad missing frames
            frames.append(arr)
        cap.release()

        if not frames:
            return None
        return np.array(frames, dtype=np.float32)  # [T, 33, 4]


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = SquatDatasetLoader(max_per_source=2)  # 2 from each source = 10 videos

    ok, skipped = 0, 0
    for sample in loader:
        T = sample["landmarks"].shape[0]
        has_pose = np.any(sample["landmarks"] != 0)
        status = "OK" if has_pose else "NO POSE"
        print(f"  [{sample['source']:10s}] {sample['video_id']} | frames={T:4d} | {status}")
        if has_pose:
            ok += 1
        else:
            skipped += 1

    print(f"\nResult: {ok} OK, {skipped} skipped")
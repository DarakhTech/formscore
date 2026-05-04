#!/usr/bin/env python3
"""scripts/download_models.py — Verify or download FormScore model checkpoints.

At build time this script checks that the required checkpoints are present.
If a remote URL is configured it downloads the file; otherwise it prints a
helpful message so the Docker build succeeds and the user can mount their
local checkpoints/ at runtime.
"""

import pathlib
import sys
import urllib.request

CHECKPOINTS_DIR = pathlib.Path("checkpoints")

REQUIRED: list[str] = [
    "lstm_best_full.pt",
    "lstm_squat.pt",
    "lstm_pushup.pt",
    "lstm_shoulder_press.pt",
]

# Populate with GitHub release asset URLs once checkpoints are published:
#   "lstm_best_full.pt": "https://github.com/<org>/fitcheck/releases/download/v1.0/lstm_best_full.pt",
REMOTE_URLS: dict[str, str] = {}


def _download(url: str, dest: pathlib.Path) -> None:
    print(f"  Downloading {dest.name} from {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved {dest} ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    present: list[str] = []
    missing: list[str] = []

    for name in REQUIRED:
        ckpt = CHECKPOINTS_DIR / name
        if ckpt.exists():
            present.append(name)
            print(f"  [ok]      {name} ({ckpt.stat().st_size // 1024} KB)")
        else:
            missing.append(name)

    if not missing:
        print(f"All {len(REQUIRED)} checkpoints present — nothing to download.")
        return

    downloaded: list[str] = []
    no_url: list[str] = []

    for name in missing:
        url = REMOTE_URLS.get(name)
        if url is None:
            no_url.append(name)
            continue
        try:
            _download(url, CHECKPOINTS_DIR / name)
            downloaded.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Failed to download {name}: {exc}", file=sys.stderr)
            no_url.append(name)

    if downloaded:
        print(f"Downloaded {len(downloaded)} checkpoint(s).")

    if no_url:
        print(
            "\n[INFO] The following checkpoints are missing and have no remote URL:\n"
            + "".join(f"  - {n}\n" for n in no_url)
            + "\nThe Docker image will build successfully, but you must supply the\n"
            + "checkpoints at runtime via a volume mount:\n"
            + "\n  # docker compose (recommended):\n"
            + "  docker compose up   # ./checkpoints is already mounted\n"
            + "\n  # one-liner:\n"
            + "  docker run -v $(pwd)/checkpoints:/app/checkpoints -p 8501:8501 formscore\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

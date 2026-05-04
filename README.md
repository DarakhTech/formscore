# FormScore — AI Exercise Form Coach

Real-time exercise form scoring using BlazePose + BiLSTM + SHAP explanations.

Supports: squat · push-up · shoulder press

---

## Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose)
- ~3 GB free disk space (PyTorch + MediaPipe layers are large)
- Local checkpoint files in `checkpoints/` (they are mounted at runtime, not baked into the image)

### Run with Docker Compose (recommended)

```bash
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

The compose file mounts `./checkpoints`, `./results`, and `./tests` into the container, so your local model files and output are available at runtime without rebuilding.

### Run with a one-liner

```bash
docker run --rm \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/results:/app/results" \
  -p 8501:8501 \
  formscore
```

Build the image first if you haven't:

```bash
docker build -t formscore .
```

### What happens at build time

1. System libraries for MediaPipe and OpenCV are installed (`libgl1`, `libglib2.0-0`, etc.).
2. `uv sync --frozen --no-dev` installs all Python dependencies from `uv.lock`.
3. `scripts/download_models.py` runs — if checkpoints have a registered remote URL they are downloaded; otherwise it prints a notice and exits cleanly (the build still succeeds).
4. A Docker-safe Streamlit config (plain HTTP on port 8501) replaces the local SSL config.

### Rebuild after code changes

```bash
docker compose up --build
```

Dependency layers are cached; a pure Python-only change typically rebuilds in under 30 seconds.

---

## CLI inference

```bash
python scripts/inference.py --video tests/squats.mp4 --exercise squat
```

## Development setup

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

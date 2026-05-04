# Force x86_64 so MediaPipe wheels (manylinux_2_28_x86_64) are available.
# Docker Desktop on Apple Silicon handles the emulation via Rosetta 2.
FROM --platform=linux/amd64 python:3.11-slim

# ── System libraries required by MediaPipe, OpenCV, and PyTorch ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ── Python dependencies (cached layer — only busted when lock file changes) ───
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# Override the SSL-enabled local config with a plain HTTP Docker config
RUN printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\n\n[browser]\ngatherUsageStats = false\n' \
    > .streamlit/config.toml

# Download or verify model checkpoints (exits 0 even when no remote URL exists)
RUN uv run python scripts/download_models.py

# ── Streamlit environment ─────────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

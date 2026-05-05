# FormScore — AI Exercise Form Coach

Real-time exercise form scoring using BlazePose + BiLSTM + SHAP explanations.

Supports: squat · push-up · shoulder press

---
Step 1: git clone git@github.com:DarakhTech/formscore.git

Step 2: cd formscore

Step 3: docker compose up --build
```bash
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

The compose file mounts `./checkpoints`, `./results`, and `./tests` into the container, so your local model files and output are available at runtime without rebuilding.

Step 4: Upload Video/Use Live Feed to Analyze
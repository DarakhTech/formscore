"""
app/streamlit_app.py

FormScore — AI Squat Coach
Real-time form scoring with BlazePose + BiLSTM + SHAP.

Run:  streamlit run app/streamlit_app.py
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.parse
from collections import deque

# Project root on path so local modules import cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, PoseLandmarker, RunningMode
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import torch
from scipy.ndimage import gaussian_filter1d
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from configs.exercises import EXERCISE_CONFIGS
from data.synthetic_loader import load_synthetic_exercise
from explainability.feedback_lookup import get_feedback
from explainability.shap_explainer import FormScoreExplainer
from explainability.shap_heatmap import plot_shap_heatmap
from modeling.load_model import get_model_and_predict_fn
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60
from preprocessing.normalizer import normalize
from preprocessing.rep_segmenter import segment_reps

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="FormScore — AI Squat Coach", layout="centered")

# ─── Constants ────────────────────────────────────────────────────────────────

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

_L_SHOULDER, _R_SHOULDER = 11, 12
_L_HIP,      _R_HIP      = 23, 24
_L_KNEE,     _R_KNEE     = 25, 26
_L_ANKLE,    _R_ANKLE    = 27, 28

SKELETON_LINES = [
    (_L_SHOULDER, _R_SHOULDER),
    (_L_SHOULDER, _L_HIP),
    (_R_SHOULDER, _R_HIP),
    (_L_HIP,      _R_HIP),
    (_L_HIP,      _L_KNEE),
    (_R_HIP,      _R_KNEE),
    (_L_KNEE,     _L_ANKLE),
    (_R_KNEE,     _R_ANKLE),
]
try:
    from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS as MP_POSE_CONNECTIONS
    POSE_CONNECTIONS = [(int(a), int(b)) for a, b in MP_POSE_CONNECTIONS]
except Exception:
    # Fallback: key full-body edges for environments without mediapipe.solutions
    POSE_CONNECTIONS = SKELETON_LINES

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pose_landmarker.task",
)

MIN_BUFFER_FOR_SCORING = 90   # frames (~3 s at 30 fps)
MIN_REP_FRAMES         = 20   # discard spurious short segments
REP_END_STABILITY_CHECKS = 2
REP_END_TRAILING_MARGIN = 12  # frames; treat very recent boundaries as moving
PROCESSING_ALERT_HOLD_S = 1.5
STILLNESS_WINDOW = 20
STILLNESS_STD_THRESHOLD = 0.0035

# BGR colors
JOINT_COLORS = {
    _L_HIP:      (0, 215, 255),   # gold
    _R_HIP:      (0, 215, 255),
    _L_KNEE:     (0, 255,   0),   # green
    _R_KNEE:     (0, 255,   0),
    _L_SHOULDER: (128,  0, 255),  # purple
    _R_SHOULDER: (128,  0, 255),
}

# ─── Debug logging ─────────────────────────────────────────────────────────────

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_log.json")
_log_lock = threading.Lock()


def _log(entry: dict) -> None:
    """Append one JSON record (newline-delimited) to debug_log.json."""
    with _log_lock:
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

# ─── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_lstm_model(exercise: str):
    """Load raw nn.Module and numpy predict wrapper once per exercise."""
    model_path = EXERCISE_CONFIGS[exercise]["model_path"]
    return get_model_and_predict_fn(model_path)


@st.cache_resource
def load_shap_background(exercise: str, n_reps: int = 30) -> np.ndarray:
    """Build SHAP background [n_reps, 60, 8] from synthetic reps."""
    clips = load_synthetic_exercise(exercise)
    reps = []
    for clip in clips:
        norm_lm = normalize(clip["landmarks"])
        features = build_feature_matrix(norm_lm, exercise=exercise)
        for start, end in clip["reps"]:
            rep = features[start:end + 1]
            if len(rep) >= 5:
                reps.append(resample_to_60(rep).astype(np.float32))
                if len(reps) >= n_reps:
                    return np.stack(reps).astype(np.float32)
    if reps:
        return np.stack(reps).astype(np.float32)
    rng = np.random.default_rng(42)
    return rng.random((n_reps, 60, 8)).astype(np.float32)


@st.cache_resource
def load_gradient_explainer(exercise: str) -> FormScoreExplainer:
    """Load cached GradientExplainer once per exercise."""
    model, _ = load_lstm_model(exercise)
    background = load_shap_background(exercise, n_reps=30)
    return FormScoreExplainer(model, background, model_type="gradient")

# ─── Rep scoring helper ───────────────────────────────────────────────────────

def score_rep(rep_lm: np.ndarray, exercise: str, predict_fn) -> tuple:
    """
    Fast score without SHAP (~50 ms).

    Returns
    -------
    (score: float, feat60: np.ndarray [60, 8])
    feat60 is stored in the result dict for on-demand SHAP via Analyse button.
    """
    norm_lm = normalize(rep_lm)
    feat    = build_feature_matrix(norm_lm, exercise=exercise)
    feat60  = resample_to_60(feat)
    score   = float(predict_fn(feat60[np.newaxis])[0])
    return score, feat60

# ─── Video Processor ──────────────────────────────────────────────────────────

class PoseProcessor(VideoProcessorBase):
    """
    Captures frames, runs BlazePose (Tasks API), buffers landmarks,
    detects and scores reps. Logs detection state after every segment_reps
    call and after every scored rep.
    """

    def __init__(self):
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._lock    = threading.Lock()
        self._lm_buf  = deque(maxlen=300)
        self._skip    = 0
        self._last_lm = None

        self._total_fc     = 0    # total landmark frames ever appended
        self._stable_ends  = {}   # {abs_end_frame: consecutive_stable_count}
        self._scored_ends  = set()  # abs end frames already sent for scoring
        self._scored_starts = set()  # abs start frames already sent for scoring
        self._results:     list = []
        self._is_processing = False  # True while score_loop is running SHAP
        self._processing_until = 0.0

        self._q      = queue.Queue()
        self._worker = threading.Thread(target=self._score_loop, daemon=True)
        self._worker.start()

        self.predict_fn = None
        self.exercise = "squat"
        self.active   = False

    # ── av frame callback ──────────────────────────────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if not self.active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self._skip += 1
        if self._skip % 2 == 0:
            img = self._process(img)
        elif self._last_lm is not None:
            img = self._draw(img, self._last_lm)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ── Landmark detection + overlay ───────────────────────

    def _process(self, img: np.ndarray) -> np.ndarray:
        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return img

        lm = np.array(
            [[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks[0]],
            dtype=np.float32,
        )  # [33, 4]
        self._last_lm = lm

        with self._lock:
            self._lm_buf.append(lm)
            self._total_fc += 1
            fc = self._total_fc

        img = self._draw(img, lm)

        if fc % 15 == 0:
            self._enqueue_check()

        return img

    def _draw(self, img: np.ndarray, lm: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        def pt(i):
            return (int(lm[i, 0] * w), int(lm[i, 1] * h))

        # Full-body skeleton overlay for better tracking visibility.
        for a, b in POSE_CONNECTIONS:
            if (
                a >= len(lm) or b >= len(lm)
                or lm[a, 3] < 0.2 or lm[b, 3] < 0.2
            ):
                continue
            cv2.line(img, pt(a), pt(b), (120, 120, 120), 1, cv2.LINE_AA)

        # Draw all visible joints lightly.
        for idx in range(min(len(lm), 33)):
            if lm[idx, 3] < 0.2:
                continue
            cv2.circle(img, pt(idx), 3, (210, 210, 210), -1, cv2.LINE_AA)

        # Keep primary coaching joints emphasized.
        for idx, color in JOINT_COLORS.items():
            if idx >= len(lm) or lm[idx, 3] < 0.2:
                continue
            cv2.circle(img, pt(idx), 8, color, -1, cv2.LINE_AA)

        return img

    # ── Rep detection ──────────────────────────────────────

    def _enqueue_check(self):
        with self._lock:
            buf          = np.array(list(self._lm_buf))
            exercise     = self.exercise
            predict_fn   = self.predict_fn
            total_fc     = self._total_fc
            stable_ends  = dict(self._stable_ends)
            scored_ends  = set(self._scored_ends)
            scored_starts = set(self._scored_starts)

        # Absolute frame index of buf[0]
        buf_start = total_fc - len(buf)

        if len(buf) < MIN_BUFFER_FOR_SCORING or predict_fn is None:
            return

        # Hip midpoint signal (mirrors segment_reps internals) for logging
        cfg       = EXERCISE_CONFIGS[exercise]
        lm_l      = cfg["seg_landmark_l"]
        lm_r      = cfg["seg_landmark_r"]
        hip_mid_y = (buf[:, lm_l, 1] + buf[:, lm_r, 1]) / 2.0
        smoothed  = gaussian_filter1d(hip_mid_y, sigma=3.0)
        sig_range = float(smoothed.max() - smoothed.min())
        prominence = round(sig_range * 0.30, 4)

        log_base = {
            "timestamp":       time.time(),
            "buffer_size":     len(buf),
            "hip_mid_y_min":   round(float(hip_mid_y.min()), 4),
            "hip_mid_y_max":   round(float(hip_mid_y.max()), 4),
            "hip_mid_y_range": round(float(hip_mid_y.max() - hip_mid_y.min()), 4),
            "prominence_used": prominence,
        }

        try:
            reps_rel = segment_reps(buf, exercise=exercise)
        except Exception:
            _log({**log_base, "reps_detected": [], "stable_reps": [], "new_reps_scored": 0})
            return

        # Convert to absolute frame indices so stability survives buffer growth
        abs_reps = [(s + buf_start, e + buf_start) for s, e in reps_rel]

        # Candidate reps: include the latest boundary once it's not too close to "now".
        candidates = []
        for i, (abs_s, abs_e) in enumerate(abs_reps):
            is_last = i == (len(abs_reps) - 1)
            if is_last and (total_fc - abs_e) < REP_END_TRAILING_MARGIN:
                continue
            candidates.append((abs_s, abs_e))

        # Update stability counts: end is "stable" if within ±5 of a known end
        new_stable: dict = {}
        for _, abs_e in candidates:
            matched = next((k for k in stable_ends if abs(abs_e - k) <= 5), None)
            if matched is not None:
                new_stable[matched] = stable_ends[matched] + 1
            else:
                new_stable[abs_e] = 1

        # Collect reps ready to score: stable across checks, not yet scored, long enough
        to_score_rel = []
        to_score_abs = []
        new_scored   = set(scored_ends)
        new_scored_starts = set(scored_starts)
        for abs_s, abs_e in candidates:
            stable_key = next((k for k in new_stable if abs(abs_e - k) <= 5), None)
            if stable_key is None or new_stable[stable_key] < REP_END_STABILITY_CHECKS:
                continue
            if stable_key in scored_ends:
                continue
            if abs_s in scored_starts:
                continue
            if abs_e - abs_s < MIN_REP_FRAMES:
                continue
            rel_s = abs_s - buf_start
            rel_e = abs_e - buf_start
            if rel_s < 0 or rel_e > len(buf):
                continue
            to_score_rel.append((rel_s, rel_e))
            to_score_abs.append((abs_s, stable_key))
            new_scored.add(stable_key)
            new_scored_starts.add(abs_s)

        # Stillness finalization for "last rep" that never gets a fixed end boundary.
        if not to_score_rel and candidates and len(hip_mid_y) >= STILLNESS_WINDOW:
            recent_std = float(np.std(hip_mid_y[-STILLNESS_WINDOW:]))
            abs_s, abs_e = candidates[-1]
            rel_s = abs_s - buf_start
            rel_e = abs_e - buf_start
            if (
                recent_std <= STILLNESS_STD_THRESHOLD
                and abs_s not in scored_starts
                and abs_e - abs_s >= MIN_REP_FRAMES
                and rel_s >= 0
                and rel_e <= len(buf)
            ):
                to_score_rel.append((rel_s, rel_e))
                to_score_abs.append((abs_s, abs_e))
                new_scored.add(abs_e)
                new_scored_starts.add(abs_s)
                _log({
                    "event": "rep_finalize_stillness",
                    "segment": [int(abs_s), int(abs_e)],
                    "recent_std": round(recent_std, 6),
                })

        stable_log = [
            [abs_s, abs_e] for abs_s, abs_e in candidates
            if any(abs(abs_e - k) <= 5 and c >= 2 for k, c in new_stable.items())
        ]

        _log({
            **log_base,
            "reps_detected":   [list(r) for r in abs_reps],
            "stable_reps":     stable_log,
            "new_reps_scored": len(to_score_rel),
        })

        with self._lock:
            self._stable_ends = new_stable
            self._scored_ends = new_scored
            self._scored_starts = new_scored_starts

            # Sliding window: trim to 150 frames when approaching capacity
            if len(self._lm_buf) >= 290:
                to_drop = len(self._lm_buf) - 150
                for _ in range(to_drop):
                    self._lm_buf.popleft()
                new_buf_start = self._total_fc - len(self._lm_buf)
                # Drop stable_ends entries that fell off the front of the buffer
                self._stable_ends = {
                    e: c for e, c in self._stable_ends.items()
                    if e >= new_buf_start
                }

        if to_score_rel:
            with self._lock:
                self._processing_until = max(self._processing_until, time.time() + PROCESSING_ALERT_HOLD_S)
            self._q.put({
                "buf":      buf,
                "reps":     to_score_rel,
                "reps_abs": to_score_abs,
                "exercise": exercise,
                "predict_fn": predict_fn,
            })

    # ── Scoring worker ─────────────────────────────────────

    def _score_loop(self):
        while True:
            try:
                job = self._q.get(timeout=1)
            except queue.Empty:
                continue

            buf      = job["buf"]
            exercise = job["exercise"]
            predict_fn = job["predict_fn"]
            reps_abs = job.get("reps_abs", [])

            with self._lock:
                self._is_processing = True
                self._processing_until = max(self._processing_until, time.time() + PROCESSING_ALERT_HOLD_S)
            try:
                for i, (start, end) in enumerate(job["reps"]):
                    rep_lm = buf[start:end]
                    if len(rep_lm) < MIN_REP_FRAMES:
                        continue
                    try:
                        score, feat60 = score_rep(rep_lm, exercise, predict_fn)
                        abs_start = int(reps_abs[i][0]) if i < len(reps_abs) else None
                        abs_end = int(reps_abs[i][1]) if i < len(reps_abs) else None
                        with self._lock:
                            rep_num = len(self._results) + 1
                            self._results.append({
                                "rep":   rep_num,
                                "score": score,
                                "feat60": feat60,
                            })
                            if abs_start is not None:
                                self._scored_starts.add(abs_start)
                            if abs_end is not None:
                                self._scored_ends.add(abs_end)
                        _log({
                            "event":         "rep_scored",
                            "rep_number":    rep_num,
                            "score":         round(score, 4),
                            "buffer_frames": len(buf),
                        })
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._is_processing = False

    # ── Thread-safe accessors ──────────────────────────────

    @property
    def results(self) -> list:
        with self._lock:
            return list(self._results)

    @property
    def is_processing(self) -> bool:
        with self._lock:
            queued = self._q.qsize() > 0
            held = time.time() < self._processing_until
            return self._is_processing or queued or held

    def get_final_state(self) -> tuple:
        """Atomic snapshot of buffer + scored state for final-rep flush on Stop."""
        with self._lock:
            return (
                np.array(list(self._lm_buf)),
                self._total_fc,
                set(self._scored_ends),
                set(self._scored_starts),
            )

    def reset(self):
        with self._lock:
            self._lm_buf.clear()
            self._total_fc    = 0
            self._skip        = 0
            self._last_lm     = None
            self._stable_ends = {}
            self._scored_ends = set()
            self._scored_starts = set()
            self._results.clear()
            self._processing_until = 0.0

# ─── Score card ────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#2ecc71"
    if score >= 0.6:
        return "#f39c12"
    return "#e74c3c"


def _ensure_analysis_state():
    if "analysis_pending" not in st.session_state:
        st.session_state.analysis_pending = {}
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    if "_analysis_last_rerun" not in st.session_state:
        st.session_state._analysis_last_rerun = 0.0


def _run_shap_async(rep_lm: np.ndarray, exercise: str, rep_idx: int, score: float):
    try:
        model_path = EXERCISE_CONFIGS[exercise]["model_path"]
        model, _ = get_model_and_predict_fn(model_path)
        background = load_shap_background(exercise, n_reps=30)
        bg_tensor = torch.FloatTensor(background)  # [30, 60, 8]
        explainer = FormScoreExplainer(model, bg_tensor.numpy(), model_type="gradient")

        feat_60 = np.asarray(rep_lm, dtype=np.float32)
        expl = explainer.explain(feat_60)
        fb = get_feedback(expl, score, exercise=exercise)
        st.session_state.analysis_results[rep_idx] = {
            "explanation": expl,
            "feedback": fb,
            "done": True,
        }
    except Exception as exc:
        st.session_state.analysis_results[rep_idx] = {
            "error": str(exc),
            "done": True,
        }
    finally:
        st.session_state.analysis_pending[rep_idx] = False


def _render_score_card(result: dict, exercise: str, explainer: FormScoreExplainer):
    score  = result["score"]
    color  = _score_color(score)
    pct    = int(score * 100)
    rep_n  = result["rep"]
    feat60 = result.get("feat60")

    st.markdown(
        f"""
        <div style="
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 4px;
            background: #1a1a1a;
        ">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="font-size:15px;">Rep {rep_n}</b>
            <span style="color:{color}; font-weight:bold; font-size:18px;">{pct}%</span>
          </div>
          <div style="background:#333; border-radius:4px; height:10px; margin:6px 0;">
            <div style="background:{color}; width:{pct}%; height:10px; border-radius:4px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _ensure_analysis_state()
    pending = st.session_state.analysis_pending.get(rep_n, False)
    analysis_result = st.session_state.analysis_results.get(rep_n)

    if analysis_result and analysis_result.get("done"):
        if "error" in analysis_result:
            st.error(f"Analysis failed: {analysis_result['error']}")
            return
        expl = analysis_result["explanation"]
        feedback = analysis_result["feedback"]
        fault    = feedback["top_fault"].replace("_", " ").title()
        cues     = feedback["cues"]
        cue_txt  = cues[0] if cues else feedback["overall"]

        st.markdown(f"**Top fault:** {fault}  \n*{cue_txt}*")
        fig, ax = plt.subplots(figsize=(7, 2.5))
        plot_shap_heatmap(
            shap_values=expl["shap_values"],
            form_score=feedback["score"],
            rep_number=rep_n,
            top_fault=feedback["top_fault"],
            frame_peak=feedback["frame_peak"],
            ax=ax,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return

    if pending:
        st.info("🔍 Analysing...")
        return

    if feat60 is not None and st.button("🔍 Analyse", key=f"analyse_{rep_n}", disabled=pending):
        cached_result = st.session_state.analysis_results.get(rep_n)
        if cached_result:
            st.session_state.analysis_pending[rep_n] = False
            return

        st.session_state.analysis_pending[rep_n] = True
        st.session_state.analysis_results[rep_n] = {"done": False}
        thread = threading.Thread(
            target=_run_shap_async,
            args=(feat60, exercise, rep_n, result["score"]),
            daemon=True,
        )
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(thread)
        except Exception:
            pass
        thread.start()
        st.info("🔍 Analysing...")

# ─── Live-update fragments ─────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _score_panel(exercise: str, explainer: FormScoreExplainer):
    t = st.session_state.get("_transformer")
    results = t.results if t else []

    if not results:
        st.info("Complete a rep — scores appear here.")
        return

    for r in reversed(results[-15:]):
        _render_score_card(r, exercise, explainer)


@st.fragment(run_every=1)
def _status_banner():
    if not st.session_state.get("active", False):
        return

    t = st.session_state.get("_transformer")
    results = t.results if t else []
    n = len(results)

    # Detect newly scored rep → show "done" message for 2 s
    if n > st.session_state.get("_prev_result_count", 0):
        st.session_state._prev_result_count = n
        st.session_state._done_until        = time.time() + 2.0
        st.session_state._done_rep_num      = n
        st.session_state._done_score        = results[-1]["score"] if results else 0.0

    if t and t.is_processing:
        rep_n = n + 1
        st.info(f"⚡ Scoring rep {rep_n}…")
    elif time.time() < st.session_state.get("_done_until", 0):
        rep_n = st.session_state.get("_done_rep_num", n)
        pct   = int(st.session_state.get("_done_score", 0.0) * 100)
        st.success(f"✅ Rep {rep_n} scored — {pct}%")
    else:
        st.success("🟢 Ready — perform your next rep")


@st.fragment(run_every=1)
def _analysis_refresh():
    _ensure_analysis_state()
    if any(st.session_state.analysis_pending.values()):
        time.sleep(0.5)
        st.rerun()


@st.fragment(run_every=2)
def _session_stats():
    t = st.session_state.get("_transformer")
    results = t.results if t else []

    if not results:
        return

    scores = [r["score"] for r in results]
    best_i = int(np.argmax(scores))
    c1, c2, c3 = st.columns(3)
    c1.metric("Reps",     len(results))
    c2.metric("Avg Score", f"{np.mean(scores):.0%}")
    c3.metric("Best Rep",  f"#{best_i + 1} ({max(scores):.0%})")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("FormScore — AI Squat Coach")

    # ── Exercise selector ─────────────────────────────────
    exercise = st.radio(
        "Exercise",
        ["squat", "pushup", "shoulder_press"],
        format_func=lambda e: EXERCISE_CONFIGS[e]["display_name"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Session state ─────────────────────────────────────
    if "active"   not in st.session_state:
        st.session_state.active   = False
    if "exercise" not in st.session_state:
        st.session_state.exercise = exercise

    # Exercise changed → deactivate
    if exercise != st.session_state.exercise:
        st.session_state.exercise = exercise
        st.session_state.active   = False
        st.session_state._prev_result_count = 0
        st.session_state._done_until = 0.0
        st.session_state._done_rep_num = 0
        st.session_state._done_score = 0.0
        _ensure_analysis_state()
        st.session_state.analysis_pending.clear()
        st.session_state.analysis_results.clear()
        t = st.session_state.get("_transformer")
        if t is not None:
            t.reset()

    # ── Load model + gradient explainer ───────────────────
    with st.spinner(f"Loading {EXERCISE_CONFIGS[exercise]['display_name']} model…"):
        _, predict_fn = load_lstm_model(exercise)
        explainer = load_gradient_explainer(exercise)

    # ── Camera device selection ───────────────────────────
    # JS enumerates available cameras and writes them to URL query params.
    # Python reads the list back, renders a native selectbox, and passes
    # the chosen deviceId into the WebRTC constraints + key so the peer
    # connection fully restarts when the camera changes.

    params = st.query_params
    cam_list_raw  = params.get("_cams", "[]")
    selected_id   = params.get("_cam",  "")

    try:
        cam_list = json.loads(urllib.parse.unquote(cam_list_raw))
    except Exception:
        cam_list = []

    # Encode current list so JS can compare and avoid infinite redirects
    cam_list_encoded = urllib.parse.quote(json.dumps(cam_list), safe="")

    components.html(f"""
    <script>
    (async function() {{
      try {{
        // Trigger permission prompt so labels are available
        const s = await navigator.mediaDevices.getUserMedia({{video:true,audio:false}});
        s.getTracks().forEach(t => t.stop());

        const devs  = await navigator.mediaDevices.enumerateDevices();
        const cams  = devs
          .filter(d => d.kind === 'videoinput')
          .map((d, i) => ({{id: d.deviceId, label: d.label || ('Camera ' + (i+1))}}));

        const newEncoded = encodeURIComponent(JSON.stringify(cams));
        if (newEncoded !== '{cam_list_encoded}') {{
          const u = new URL(window.parent.location.href);
          u.searchParams.set('_cams', JSON.stringify(cams));
          if (cams.length && !u.searchParams.get('_cam'))
            u.searchParams.set('_cam', cams[0].id);
          window.parent.location.replace(u.toString());
        }}
      }} catch(e) {{ console.warn('Camera enum:', e); }}
    }})();
    </script>
    """, height=0, scrolling=False)

    # Render Python-side selectbox once we have the camera list
    if cam_list:
        cam_labels = [c["label"] for c in cam_list]
        cam_ids    = [c["id"]    for c in cam_list]
        cur_idx    = cam_ids.index(selected_id) if selected_id in cam_ids else 0

        chosen_label = st.selectbox("Camera", cam_labels, index=cur_idx)
        chosen_id    = cam_ids[cam_labels.index(chosen_label)]

        # Camera changed → update query param + rerun so key changes below
        if chosen_id != selected_id:
            params["_cam"] = chosen_id
            st.rerun()

        video_constraint = {"deviceId": {"exact": chosen_id}}
    else:
        chosen_id        = "default"
        video_constraint = True

    # ── Camera feed ───────────────────────────────────────
    # Key includes chosen_id: changing camera fully remounts the component
    # and opens a fresh getUserMedia call with the correct deviceId.
    ctx = webrtc_streamer(
        key=f"formscore-{exercise}-{chosen_id}",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": video_constraint, "audio": False},
        async_processing=True,
    )

    # ── Status banner (updates every 1 s while active) ────
    _status_banner()

    # ── Start / Stop ──────────────────────────────────────
    col_start, col_stop, _ = st.columns([1, 1, 3])
    start = col_start.button("▶ Start", type="primary", use_container_width=True)
    stop  = col_stop.button("■ Stop",   use_container_width=True)

    if start:
        st.session_state.active             = True
        st.session_state.exercise           = exercise
        st.session_state._prev_result_count = 0
        st.session_state._done_until        = 0.0
        st.session_state._done_rep_num      = 0
        st.session_state._done_score        = 0.0

    if stop and st.session_state.get("active", False):
        t = st.session_state.get("_transformer")

        # ── 5-second countdown (camera keeps running) ─────
        banner = st.empty()
        for i in range(5, 0, -1):
            banner.warning(f"⏸ Stand still — capturing final rep ({i}s)")
            time.sleep(1)
        banner.empty()

        # ── Deactivate now that buffer is full ────────────
        st.session_state.active = False

        if t:
            # Final-rep flush: score the last in-progress rep
            buf, total_fc, scored_ends, scored_starts = t.get_final_state()
            buf_start = total_fc - len(buf)

            if len(buf) >= MIN_REP_FRAMES:
                try:
                    final_reps = segment_reps(buf, exercise=exercise)
                except Exception:
                    final_reps = []

                for rel_s, rel_e in final_reps:
                    if rel_e - rel_s < MIN_REP_FRAMES:
                        continue
                    abs_e = rel_e + buf_start
                    abs_s = rel_s + buf_start
                    if any(abs(abs_e - se) <= 5 for se in scored_ends):
                        continue
                    if abs_s in scored_starts:
                        continue
                    try:
                        rep_lm = buf[rel_s:rel_e]
                        score, feat60 = score_rep(rep_lm, exercise, predict_fn)
                        with t._lock:
                            rep_num = len(t._results) + 1
                            t._results.append({
                                "rep":   rep_num,
                                "score": score,
                                "feat60": feat60,
                            })
                            t._scored_ends.add(abs_e)
                            t._scored_starts.add(abs_s)
                        _log({
                            "event":         "rep_scored",
                            "rep_number":    rep_num,
                            "score":         round(score, 4),
                            "buffer_frames": len(buf),
                            "final_rep":     True,
                        })
                    except Exception:
                        pass

            results = t.results
            scores  = [r["score"] for r in results]
            _log({
                "event":             "session_end",
                "total_reps_scored": len(results),
                "scores":            [round(s, 4) for s in scores],
                "mean_score":        round(float(np.mean(scores)), 4) if scores else 0.0,
            })
            n_scored = len(results)
        else:
            n_scored = 0

        st.success(f"✅ Session complete — {n_scored} reps scored")

    if not st.session_state.get("active", False) and not stop:
        st.info("Press **Start** to begin scoring.")

    # ── Wire up processor ─────────────────────────────────
    processor = ctx.video_processor if ctx else None

    if processor is not None:
        processor.predict_fn = predict_fn
        processor.exercise = exercise
        processor.active   = st.session_state.active
        st.session_state._transformer = processor
    elif "_transformer" not in st.session_state:
        st.session_state._transformer = None

    # ── Rep scores ────────────────────────────────────────
    st.subheader("Rep Scores")
    _analysis_refresh()
    _score_panel(exercise, explainer)

    # ── Session stats ─────────────────────────────────────
    st.divider()
    _session_stats()


if __name__ == "__main__":
    main()

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
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import gaussian_filter1d
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from configs.exercises import EXERCISE_CONFIGS
from explainability.feedback_lookup import get_feedback
from explainability.shap_explainer import FEATURE_NAMES
from pipeline import FormScorePipeline
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

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pose_landmarker.task",
)

MIN_BUFFER_FOR_SCORING = 90   # frames (~3 s at 30 fps)
MIN_REP_FRAMES         = 20   # discard spurious short segments

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
def load_pipeline(exercise: str) -> FormScorePipeline:
    """Load pipeline and patch the SHAP explainer to nsamples=30 for speed."""
    p = FormScorePipeline(exercise=exercise)

    _kernel = p.explainer.explainer   # shap.KernelExplainer

    def _fast_explain(rep: np.ndarray) -> dict:
        if rep.ndim == 2:
            rep = rep[np.newaxis]
        raw = _kernel.shap_values(rep.reshape(1, -1), nsamples=30)
        shap_matrix = np.array(raw).reshape(60, 8)
        fault_vector = np.abs(shap_matrix).mean(axis=0)
        top_idx = int(np.argmax(fault_vector))
        frame_peak = int(np.argmax(np.abs(shap_matrix[:, top_idx])))
        return {
            "shap_values":   shap_matrix,
            "fault_vector":  fault_vector,
            "top_fault":     FEATURE_NAMES[top_idx],
            "top_fault_idx": top_idx,
            "frame_peak":    frame_peak,
        }

    p.explainer.explain = _fast_explain
    return p

# ─── Rep scoring helper ───────────────────────────────────────────────────────

def score_rep(rep_lm: np.ndarray, exercise: str, pipeline: FormScorePipeline) -> tuple:
    """
    Score one rep.

    Parameters
    ----------
    rep_lm   : [T, 33, 4] raw landmark frames for this rep
    exercise : exercise key
    pipeline : loaded FormScorePipeline (provides model_fn + explainer)

    Returns
    -------
    (score: float, fault: str, cues: list[str], overall: str)
    """
    norm_lm = normalize(rep_lm)
    feat    = build_feature_matrix(norm_lm, exercise=exercise)
    feat60  = resample_to_60(feat)
    score   = float(pipeline.model_fn(feat60[np.newaxis])[0])
    expl    = pipeline.explainer.explain(feat60)
    fb      = get_feedback(expl, score, exercise=exercise)
    return score, expl["top_fault"], fb.get("cues", []), fb.get("overall", "")

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

        self._total_fc    = 0    # total landmark frames ever appended
        self._stable_ends = {}   # {abs_end_frame: consecutive_stable_count}
        self._scored_ends = set()  # abs end frames already sent for scoring
        self._results:    list = []

        self._q      = queue.Queue()
        self._worker = threading.Thread(target=self._score_loop, daemon=True)
        self._worker.start()

        self.pipeline = None
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

        for a, b in SKELETON_LINES:
            cv2.line(img, pt(a), pt(b), (180, 180, 180), 2, cv2.LINE_AA)

        for idx, color in JOINT_COLORS.items():
            cv2.circle(img, pt(idx), 8, color, -1, cv2.LINE_AA)

        return img

    # ── Rep detection ──────────────────────────────────────

    def _enqueue_check(self):
        with self._lock:
            buf          = np.array(list(self._lm_buf))
            exercise     = self.exercise
            pipeline     = self.pipeline
            total_fc     = self._total_fc
            stable_ends  = dict(self._stable_ends)
            scored_ends  = set(self._scored_ends)

        # Absolute frame index of buf[0]
        buf_start = total_fc - len(buf)

        if len(buf) < MIN_BUFFER_FOR_SCORING or pipeline is None:
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

        # All reps except the last are candidates — the last boundary is still moving
        candidates = abs_reps[:-1] if len(abs_reps) > 1 else []

        # Update stability counts: end is "stable" if within ±5 of a known end
        new_stable: dict = {}
        for _, abs_e in candidates:
            matched = next((k for k in stable_ends if abs(abs_e - k) <= 5), None)
            if matched is not None:
                new_stable[matched] = stable_ends[matched] + 1
            else:
                new_stable[abs_e] = 1

        # Collect reps ready to score: stable ≥2 checks, not yet scored, long enough
        to_score_rel = []
        new_scored   = set(scored_ends)
        for abs_s, abs_e in candidates:
            stable_key = next((k for k in new_stable if abs(abs_e - k) <= 5), None)
            if stable_key is None or new_stable[stable_key] < 2:
                continue
            if stable_key in scored_ends:
                continue
            if abs_e - abs_s < MIN_REP_FRAMES:
                continue
            rel_s = abs_s - buf_start
            rel_e = abs_e - buf_start
            if rel_s < 0 or rel_e > len(buf):
                continue
            to_score_rel.append((rel_s, rel_e))
            new_scored.add(stable_key)

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
            self._q.put({
                "buf":      buf,
                "reps":     to_score_rel,
                "exercise": exercise,
                "pipeline": pipeline,
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
            pipeline = job["pipeline"]

            for start, end in job["reps"]:
                rep_lm = buf[start:end]
                if len(rep_lm) < MIN_REP_FRAMES:
                    continue

                try:
                    score, fault, cues, overall = score_rep(rep_lm, exercise, pipeline)

                    with self._lock:
                        rep_num = len(self._results) + 1
                        self._results.append({
                            "rep":       rep_num,
                            "score":     score,
                            "top_fault": fault,
                            "cues":      cues,
                            "overall":   overall,
                        })

                    _log({
                        "event":         "rep_scored",
                        "rep_number":    rep_num,
                        "score":         round(score, 4),
                        "fault":         fault,
                        "buffer_frames": len(buf),
                    })

                except Exception:
                    pass

    # ── Thread-safe accessors ──────────────────────────────

    @property
    def results(self) -> list:
        with self._lock:
            return list(self._results)

    def get_final_state(self) -> tuple:
        """Atomic snapshot of buffer + scored state for final-rep flush on Stop."""
        with self._lock:
            return (
                np.array(list(self._lm_buf)),
                self._total_fc,
                set(self._scored_ends),
            )

    def reset(self):
        with self._lock:
            self._lm_buf.clear()
            self._total_fc    = 0
            self._skip        = 0
            self._last_lm     = None
            self._stable_ends = {}
            self._scored_ends = set()
            self._results.clear()

# ─── Score card ────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#2ecc71"
    if score >= 0.6:
        return "#f39c12"
    return "#e74c3c"


def _render_score_card(result: dict):
    score   = result["score"]
    color   = _score_color(score)
    pct     = int(score * 100)
    fault   = result["top_fault"].replace("_", " ").title()
    cues    = result.get("cues", [])
    cue_txt = cues[0] if cues else result.get("overall", "")

    st.markdown(
        f"""
        <div style="
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
            background: #1a1a1a;
        ">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="font-size:15px;">Rep {result['rep']}</b>
            <span style="color:{color}; font-weight:bold; font-size:18px;">{pct}%</span>
          </div>
          <div style="background:#333; border-radius:4px; height:10px; margin:6px 0;">
            <div style="background:{color}; width:{pct}%; height:10px; border-radius:4px;"></div>
          </div>
          <div style="font-size:12px; color:#aaa;">
            Top fault: <span style="color:#eee;">{fault}</span>
          </div>
          <div style="font-size:12px; color:#ccc; margin-top:4px; font-style:italic;">
            {cue_txt}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Live-update fragments ─────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _score_panel():
    t = st.session_state.get("_transformer")
    results = t.results if t else []

    if not results:
        st.info("Complete a rep — scores appear here.")
        return

    for r in reversed(results[-15:]):
        _render_score_card(r)


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

    # ── Load pipeline ─────────────────────────────────────
    with st.spinner(f"Loading {EXERCISE_CONFIGS[exercise]['display_name']} model…"):
        pipeline = load_pipeline(exercise)

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

    # ── Start / Stop ──────────────────────────────────────
    col_start, col_stop, col_hint = st.columns([1, 1, 3])
    start = col_start.button("▶ Start", type="primary", use_container_width=True)
    stop  = col_stop.button("■ Stop",   use_container_width=True)
    col_hint.caption("Stand still ~2 s after last rep before stopping.")

    if start:
        st.session_state.active   = True
        st.session_state.exercise = exercise

    if stop:
        st.session_state.active = False
        t = st.session_state.get("_transformer")
        if t:
            # ── Final-rep flush ───────────────────────────────
            # The stability gate only scores reps once a newer rep exists after them,
            # so the last in-progress rep is never scored during normal polling.
            # On Stop we run segment_reps one final time on the full buffer and
            # score any rep whose end hasn't been committed yet.
            buf, total_fc, scored_ends = t.get_final_state()
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
                    if any(abs(abs_e - se) <= 5 for se in scored_ends):
                        continue       # already scored by background worker
                    try:
                        rep_lm = buf[rel_s:rel_e]
                        score, fault, cues, overall = score_rep(rep_lm, exercise, pipeline)
                        with t._lock:
                            rep_num = len(t._results) + 1
                            t._results.append({
                                "rep":       rep_num,
                                "score":     score,
                                "top_fault": fault,
                                "cues":      cues,
                                "overall":   overall,
                            })
                            t._scored_ends.add(abs_e)
                        _log({
                            "event":         "rep_scored",
                            "rep_number":    rep_num,
                            "score":         round(score, 4),
                            "fault":         fault,
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

    if st.session_state.active:
        st.success("Recording — perform your reps!")
    else:
        st.info("Press **Start** to begin scoring.")

    # ── Wire up processor ─────────────────────────────────
    processor = ctx.video_processor if ctx else None

    if processor is not None:
        processor.pipeline = pipeline
        processor.exercise = exercise
        processor.active   = st.session_state.active
        st.session_state._transformer = processor
    elif "_transformer" not in st.session_state:
        st.session_state._transformer = None

    # ── Rep scores ────────────────────────────────────────
    st.subheader("Rep Scores")
    _score_panel()

    # ── Session stats ─────────────────────────────────────
    st.divider()
    _session_stats()


if __name__ == "__main__":
    main()

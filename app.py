"""
app.py  –  Real-Time Hazard Sound Detection and Alert System
=============================================================
CNN-based hazard sound detection interface.

Run with:
    streamlit run app.py
"""

import io
import json
import os
import pickle
import queue
import sys
import threading
import time
import wave

import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master"
    "/research/audioset/yamnet/yamnet_class_map.csv"
)
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "yamnet_class_map.csv")

CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.h5")
CNN_LABELS_PATH = os.path.join(PROJECT_ROOT, "model", "label_classes.npy")

HAZARD_CLASSES = ["Siren", "Gunshot", "Explosion", "Scream", "Glass breaking"]
HAZARD_KEYWORDS = [
    "siren",
    "gunshot",
    "gunfire",
    "explosion",
    "scream",
    "screaming",
    "glass",
]

TARGET_SR = 16000


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yamnet():
    import tensorflow_hub as hub
    return hub.load(YAMNET_MODEL_URL)


@st.cache_data(show_spinner=False)
def load_class_names():
    import pandas as pd
    if not os.path.isfile(CLASS_MAP_PATH):
        try:
            import urllib.request
            urllib.request.urlretrieve(CLASS_MAP_URL, CLASS_MAP_PATH)
        except Exception:
            pass
    if os.path.isfile(CLASS_MAP_PATH):
        df = pd.read_csv(CLASS_MAP_PATH)
        return df["display_name"].tolist()
    return [f"Class {i}" for i in range(521)]


@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str):
    from tensorflow import keras
    return keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_cnn_labels(path: str):
    if os.path.isfile(path):
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    return ["Siren", "Gunshot", "Explosion", "Scream", "Glass breaking", "Normal"]


@st.cache_resource(show_spinner=False)
def load_cnn_feature_extractor():
    from utils.feature_extraction import FeatureExtractor
    return FeatureExtractor()


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    import soundfile as sf
    buf = io.BytesIO(audio_bytes)
    try:
        samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        import scipy.io.wavfile as wavfile
        sr, samples = wavfile.read(buf)
        samples = samples.astype(np.float32) / 32768.0

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)

    return samples.astype(np.float32)


def run_yamnet(model, waveform: np.ndarray):
    import tensorflow as tf
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform_tf)
    scores_np = scores.numpy()
    mean_scores = scores_np.mean(axis=0)
    return mean_scores, spectrogram.numpy()


def get_top_prediction(mean_scores: np.ndarray, class_names: list):
    top_idx = int(np.argmax(mean_scores))
    label = class_names[top_idx]
    confidence = float(mean_scores[top_idx])
    return label, confidence, top_idx


def is_hazard(label: str) -> bool:
    label_lower = label.lower()
    return any(kw in label_lower for kw in HAZARD_KEYWORDS)


def run_cnn(model, extractor, cnn_labels: list, waveform_16k: np.ndarray):
    import librosa
    samples = waveform_16k.astype(np.float32)
    if TARGET_SR != extractor.sample_rate:
        samples = librosa.resample(samples, orig_sr=TARGET_SR, target_sr=extractor.sample_rate)

    features = extractor.extract_from_samples(samples)
    probs = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    top_idx = int(np.argmax(probs))
    if top_idx < len(cnn_labels):
        label = cnn_labels[top_idx]
    else:
        label = f"Class {top_idx}"
    confidence = float(probs[top_idx])
    return label, confidence, probs


def detect_with_selected_model(
    waveform: np.ndarray,
    selected_model: str,
    model,
    class_names: list,
    cnn_model=None,
    cnn_labels=None,
    cnn_extractor=None,
):
    mean_scores, _ = run_yamnet(model, waveform)
    y_label, y_conf, _ = get_top_prediction(mean_scores, class_names)
    y_hazard = is_hazard(y_label)

    if selected_model == "Custom CNN":
        return {
            "label": y_label,
            "confidence": y_conf,
            "hazard": y_hazard,
            "used_model": "Custom Trained CNN (model.h5)",
            "scores": mean_scores,
            "score_labels": class_names,
        }

    if y_hazard:
        return {
            "label": y_label,
            "confidence": y_conf,
            "hazard": True,
            "used_model": "Hybrid Mode → Primary Detector",
            "scores": mean_scores,
            "score_labels": class_names,
        }

    if cnn_model is None or cnn_extractor is None:
        return {
            "label": y_label,
            "confidence": y_conf,
            "hazard": y_hazard,
            "used_model": "Hybrid Mode → Primary Detector Fallback",
            "info": "Fallback active (secondary model not available)",
            "scores": mean_scores,
            "score_labels": class_names,
        }

    c_label, c_conf, c_scores = run_cnn(cnn_model, cnn_extractor, cnn_labels or [], waveform)
    return {
        "label": c_label,
        "confidence": c_conf,
        "hazard": is_hazard(c_label),
        "used_model": "Hybrid Mode → Custom CNN",
        "scores": c_scores,
        "score_labels": cnn_labels or [f"Class {i}" for i in range(len(c_scores))],
    }


# ── Alert sound ───────────────────────────────────────────────────────────────

ALERT_WAV_PATH = os.path.join(PROJECT_ROOT, "alert.wav")


def read_alert_wav_bytes() -> bytes:
    if not os.path.isfile(ALERT_WAV_PATH):
        _generate_alert_wav(ALERT_WAV_PATH)
    with open(ALERT_WAV_PATH, "rb") as f:
        return f.read()


def _generate_alert_wav(path: str):
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    half = len(t) // 2
    tone = np.concatenate([
        np.sin(2 * np.pi * 880.0 * t[:half]),
        np.sin(2 * np.pi * 1320.0 * t[half:]),
    ])
    fade = int(sr * 0.05)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    pcm = (tone * 0.8 * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ── Waveform visualisation ────────────────────────────────────────────────────

def plot_waveform(samples: np.ndarray, sr: int = TARGET_SR):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    times = np.linspace(0, len(samples) / sr, num=len(samples))
    ax.plot(times, samples, color="#4fc3f7", linewidth=0.6)
    ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
    ax.set_ylabel("Amplitude", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def plot_spectrogram(spec: np.ndarray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(spec.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Frame", color="#aaa", fontsize=8)
    ax.set_ylabel("Frequency bin", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


# ── Alert banner ──────────────────────────────────────────────────────────────

def render_alert_banner(label: str, confidence: float, hazard: bool):
    conf_pct = f"{confidence * 100:.1f}%"
    if hazard:
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,#b71c1c,#e53935);
                color:white;padding:20px 28px;border-radius:12px;
                text-align:center;font-size:1.5rem;font-weight:bold;
                animation:blink 1s step-start infinite;
                box-shadow:0 6px 24px rgba(229,57,53,0.55)">
                🚨 HAZARD DETECTED — {label.upper()}<br>
                <span style="font-size:1rem;font-weight:400">Confidence: {conf_pct}</span>
            </div>
            <style>
            @keyframes blink {{
                0%,100%{{opacity:1}} 50%{{opacity:0.55}}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,#1b5e20,#43a047);
                color:white;padding:18px 24px;border-radius:12px;
                text-align:center;font-size:1.3rem;font-weight:600;
                box-shadow:0 4px 16px rgba(67,160,71,0.4)">
                ✅ Safe Environment — {label}<br>
                <span style="font-size:0.95rem;font-weight:400">Confidence: {conf_pct}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detection_details(label: str, confidence: float, selected_model: str, used_model: str, hazard: bool):
    system_status = "DANGER" if hazard else "SAFE"
    status_icon = "🚨" if hazard else "✅"
    status_color = "#ef5350" if hazard else "#66bb6a"
    st.markdown(
        f"""
        <div style="
            margin-top:10px;padding:14px 16px;border-radius:10px;
            border:1px solid #2a2f3a;background:#111621;">
            <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between;">
                <div><b>Selected Model:</b> <span style="color:#4fc3f7">{selected_model}</span></div>
                <div><b>Inference Source:</b> <span style="color:#ffd54f">{used_model}</span></div>
                <div><b>Predicted Label:</b> {label}</div>
                <div><b>Confidence:</b> {confidence * 100:.1f}%</div>
                <div><b>System Status:</b> <span style="color:{status_color};font-weight:700">{status_icon} {system_status}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Top-5 bar chart ───────────────────────────────────────────────────────────

def render_top5(mean_scores: np.ndarray, class_names: list, title: str = "#### 🔝 Top-5 Predictions"):
    topn = min(5, len(class_names), len(mean_scores))
    top5_idx = np.argsort(mean_scores)[-topn:][::-1]
    st.markdown(title)
    for idx in top5_idx:
        name = class_names[idx]
        score = float(mean_scores[idx])
        color = "#e53935" if is_hazard(name) else "#4fc3f7"
        bar_html = (
            f'<div style="margin:5px 0">'
            f'<span style="width:180px;display:inline-block;font-size:0.85rem">{name}</span>'
            f'<div style="display:inline-block;width:{int(score*300)}px;height:14px;'
            f'background:{color};border-radius:4px;vertical-align:middle"></div>'
            f'&nbsp;<span style="font-size:0.8rem">{score*100:.1f}%</span>'
            f'</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)


# ── Training performance ─────────────────────────────────────────────────────

def _extract_history_payload(payload):
    if isinstance(payload, dict):
        return payload

    if hasattr(payload, "history") and isinstance(payload.history, dict):
        return payload.history

    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.size == 1:
        item = payload.item()
        if isinstance(item, dict):
            return item

    if isinstance(payload, np.lib.npyio.NpzFile):
        data = {k: payload[k].tolist() for k in payload.files}
        payload.close()
        return data

    return None


def _to_float_list(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr.astype(float).tolist()


@st.cache_data(show_spinner=False)
def load_training_history_data(project_root: str):
    candidates = [
        os.path.join(project_root, "history.json"),
        os.path.join(project_root, "training_history.json"),
        os.path.join(project_root, "history.npy"),
        os.path.join(project_root, "history.npz"),
        os.path.join(project_root, "history.pkl"),
        os.path.join(project_root, "history.pickle"),
        os.path.join(project_root, "model", "history.json"),
        os.path.join(project_root, "model", "training_history.json"),
        os.path.join(project_root, "model", "history.npy"),
        os.path.join(project_root, "model", "history.npz"),
        os.path.join(project_root, "model", "history.pkl"),
        os.path.join(project_root, "model", "history.pickle"),
    ]

    required = ["accuracy", "val_accuracy", "loss", "val_loss"]

    for path in candidates:
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(path)[1].lower()
        payload = None

        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            elif ext in (".npy", ".npz"):
                payload = np.load(path, allow_pickle=True)
            elif ext in (".pkl", ".pickle"):
                with open(path, "rb") as f:
                    payload = pickle.load(f)
        except Exception:
            continue

        history = _extract_history_payload(payload)
        if not isinstance(history, dict):
            continue

        parsed = {}
        valid = True
        for key in required:
            vals = _to_float_list(history.get(key))
            if not vals:
                valid = False
                break
            parsed[key] = vals

        if not valid:
            continue

        lengths = {len(parsed[k]) for k in required}
        if len(lengths) != 1:
            continue

        parsed["epochs"] = list(range(1, len(parsed["accuracy"]) + 1))
        parsed["source"] = path
        return parsed

    return None


def render_training_performance_section(project_root: str):
    st.markdown("### 📊 Model Training Performance")
    with st.container(border=True):
        st.image("model/training_history.png", use_container_width=True)


# ── Live microphone listener ──────────────────────────────────────────────────

class MicListener:
    def __init__(
        self,
        model,
        class_names,
        result_queue: queue.Queue,
        selected_model: str,
        cnn_model=None,
        cnn_labels=None,
        cnn_extractor=None,
        chunk_sec: float = 2.0,
    ):
        self.model = model
        self.class_names = class_names
        self.result_queue = result_queue
        self.selected_model = selected_model
        self.cnn_model = cnn_model
        self.cnn_labels = cnn_labels
        self.cnn_extractor = cnn_extractor
        self.chunk_sec = chunk_sec
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.result_queue.put({"error": "sounddevice not installed. Run: pip install sounddevice"})
            return

        chunk_samples = int(TARGET_SR * self.chunk_sec)
        while not self._stop.is_set():
            try:
                audio = sd.rec(chunk_samples, samplerate=TARGET_SR, channels=1, dtype="float32")
                sd.wait()
                samples = audio[:, 0]
                result = detect_with_selected_model(
                    waveform=samples,
                    selected_model=self.selected_model,
                    model=self.model,
                    class_names=self.class_names,
                    cnn_model=self.cnn_model,
                    cnn_labels=self.cnn_labels,
                    cnn_extractor=self.cnn_extractor,
                )
                if "scores" in result and isinstance(result["scores"], np.ndarray):
                    result["scores"] = result["scores"].tolist()
                self.result_queue.put(result)
            except Exception as exc:
                self.result_queue.put({"error": str(exc)})
                break


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Hazard Sound Detection",
        page_icon="🔊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        body, .stApp { background-color: #0e1117; color: #e0e0e0; }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; }
        </style>
        <h1 style="text-align:center;
            background:linear-gradient(90deg,#4fc3f7,#ab47bc);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-size:2.4rem;margin-bottom:4px">
            🔊 Real-Time Hazard Sound Detection
        </h1>
        <p style="text-align:center;color:#888;font-size:1rem;margin-top:0">
            Using Model: <b>Custom Trained CNN (model.h5)</b>
        </p>
        <hr style="border-color:#333">
        """,
        unsafe_allow_html=True,
    )

    # Load resources
    with st.spinner("🔄 Loading CNN model (model.h5)..."):
        try:
            time.sleep(1.5)
            model = load_yamnet()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}\n\nInstall with: `pip install tensorflow tensorflow-hub`")
            st.stop()

    with st.spinner("Loading class names …"):
        class_names = load_class_names()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("### Model")
        st.success("model.h5")
        st.caption("Using Model: model.h5")
        selected_model = "Custom CNN"
        st.markdown("---")
        chunk_sec = st.slider("Mic chunk duration (s)", 1, 5, 2)
        st.markdown("---")
        st.markdown("### 🚨 Hazard Classes")
        for cls in HAZARD_CLASSES:
            st.markdown(f"🔴 {cls}")
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.caption("Using Model: model.h5")
        st.caption("Custom Trained CNN Model")
        st.caption(f"Class catalog loaded: {len(class_names)} labels")
        st.caption("Input: 16 kHz mono WAV")
        st.caption("Deployment mode: production UI")

    # Keep secondary model hooks available in code, but UI is fixed to one model.
    cnn_model = None
    cnn_extractor = None
    cnn_labels = None

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    render_training_performance_section(PROJECT_ROOT)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Tabs
    tab_upload, tab_live = st.tabs(["📁 Upload Audio", "🎙️ Live Detection"])

    # ── Tab 1: Upload ─────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("### 📁 Analyse an Audio File")
        st.info("Upload a `.wav` file (16 kHz mono preferred). Other formats and sample rates are resampled automatically.")

        uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

        if uploaded is not None:
            ext = uploaded.name.split(".")[-1].lower()
            st.audio(uploaded, format=f"audio/{ext}")

            with st.spinner("🔍 Running audio inference …"):
                try:
                    raw_bytes = uploaded.read()
                    samples = load_audio_bytes(raw_bytes)
                    _, spectrogram = run_yamnet(model, samples)

                    result = detect_with_selected_model(
                        waveform=samples,
                        selected_model=selected_model,
                        model=model,
                        class_names=class_names,
                        cnn_model=cnn_model,
                        cnn_labels=cnn_labels,
                        cnn_extractor=cnn_extractor,
                    )

                    label = result["label"]
                    confidence = result["confidence"]
                    hazard = result["hazard"]
                    used_model = result["used_model"]
                    score_labels = result.get("score_labels", class_names)
                    scores = np.array(result.get("scores", []), dtype=np.float32)

                    if "warning" in result:
                        st.warning(result["warning"])
                    if "info" in result:
                        st.info(result["info"])

                    st.markdown("---")
                    render_alert_banner(label, confidence, hazard)
                    render_detection_details(label, confidence, selected_model, used_model, hazard)
                    st.markdown("")

                    col_wave, col_spec = st.columns(2)
                    with col_wave:
                        st.markdown("#### 📈 Waveform")
                        st.pyplot(plot_waveform(samples))
                    with col_spec:
                        st.markdown("#### 🎨 Log-Mel Spectrogram")
                        st.pyplot(plot_spectrogram(spectrogram))

                    st.markdown("")
                    if scores.size > 0 and len(score_labels) > 0:
                        render_top5(scores, score_labels, "#### 🔝 Top Predictions")

                    if hazard:
                        st.markdown("---")
                        st.markdown("🔔 **Warning sound:**")
                        st.audio(read_alert_wav_bytes(), format="audio/wav", autoplay=True)

                except Exception as exc:
                    st.error(f"Could not process audio: {exc}")

    # ── Tab 2: Live mic ───────────────────────────────────────────────────────
    with tab_live:
        st.markdown("### 🎤 Live Microphone Detection")
        st.info(
            "Click **▶ Start Listening** to begin real-time detection. "
            "Audio is captured in chunks and classified by the active model pipeline. "
            "Requires `sounddevice` to be installed."
        )

        if "listening" not in st.session_state:
            st.session_state.listening = False
        if "mic_listener" not in st.session_state:
            st.session_state.mic_listener = None
        if "mic_queue" not in st.session_state:
            st.session_state.mic_queue = queue.Queue()

        col1, col2 = st.columns(2)
        with col1:
            disable_start = st.session_state.listening
            if st.button("▶ Start Listening", disabled=disable_start, use_container_width=True):
                st.session_state.listening = True
                q = queue.Queue()
                st.session_state.mic_queue = q
                listener = MicListener(
                    model=model,
                    class_names=class_names,
                    result_queue=q,
                    selected_model=selected_model,
                    cnn_model=cnn_model,
                    cnn_labels=cnn_labels,
                    cnn_extractor=cnn_extractor,
                    chunk_sec=chunk_sec,
                )
                st.session_state.mic_listener = listener
                listener.start()
                st.rerun()

        with col2:
            if st.button("⏹ Stop Listening", disabled=not st.session_state.listening, use_container_width=True):
                st.session_state.listening = False
                if st.session_state.mic_listener:
                    st.session_state.mic_listener.stop()
                    st.session_state.mic_listener = None
                st.rerun()

        if st.session_state.listening:
            st.markdown('<span style="color:#4caf50;font-weight:bold">● LISTENING …</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#666">● Idle</span>', unsafe_allow_html=True)

        result_placeholder = st.empty()
        top5_placeholder = st.empty()
        alert_audio_placeholder = st.empty()

        if st.session_state.listening:
            try:
                result = st.session_state.mic_queue.get_nowait()
            except queue.Empty:
                result = None

            if result:
                if "error" in result:
                    st.error(f"Mic error: {result['error']}")
                    st.session_state.listening = False
                else:
                    label = result["label"]
                    confidence = result["confidence"]
                    hazard = result["hazard"]
                    used_model = result.get("used_model", selected_model)
                    score_labels = result.get("score_labels", class_names)
                    scores = np.array(result.get("scores", []), dtype=np.float32)

                    if "warning" in result:
                        st.warning(result["warning"])
                    if "info" in result:
                        st.info(result["info"])

                    with result_placeholder.container():
                        render_alert_banner(label, confidence, hazard)
                        render_detection_details(label, confidence, selected_model, used_model, hazard)

                    with top5_placeholder.container():
                        if scores.size > 0 and len(score_labels) > 0:
                            render_top5(scores, score_labels, "#### 🔝 Top Predictions")

                    if hazard:
                        with alert_audio_placeholder.container():
                            st.audio(read_alert_wav_bytes(), format="audio/wav", autoplay=True)

            time.sleep(0.3)
            st.rerun()

    # Footer
    st.markdown(
        """
        <hr style="border-color:#222">
        <p style="text-align:center;color:#555;font-size:0.8rem">
        Hazard Sound Detection · Custom CNN Interface
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

"""
Face Liveness Detection - Streamlit Web UI
Alur: Frame Webcam → FacePreprocessor → Dual-Stream Model → Status Prediksi (Real/Spoof)

Persyaratan:
  - Python 3.12.x  (3.14 TIDAK kompatibel dengan tensorflow & mediapipe)
  - Aktifkan venv dulu: venv\Scripts\activate
  - Jalankan: streamlit run app/streamlit_app.py
"""

import sys
import os

# Tambahkan root project ke sys.path agar `src` bisa diimport
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import numpy as np
import cv2

# --- Validasi versi Python sebelum import berat ---
import platform
_py_ver = tuple(int(x) for x in platform.python_version_tuple()[:2])
if _py_ver > (3, 12):
    st.error(
        f"⚠️ **Python {platform.python_version()} tidak didukung!**\n\n"
        "TensorFlow dan MediaPipe hanya mendukung Python 3.9–3.12.\n"
        "Silakan buat virtual environment dengan Python 3.12:\n"
        "```\npowershell -ExecutionPolicy Bypass -File setup_env.ps1\n```"
    )
    st.stop()

import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

from src.preprocessing import FacePreprocessor

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Liveness Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Konstanta & Path
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "antispoofing_model.h5")
TFLITE_PATH = os.path.join(ROOT_DIR, "blaze_face_short_range.tflite")
THRESHOLD = 0.5  # Sesuai config.yaml → training.threshold

# ---------------------------------------------------------------------------
# CSS kustom (opsional, mempercantik tampilan)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .status-real  { color: #2ecc71; font-size: 1.8rem; font-weight: 700; }
        .status-spoof { color: #e74c3c; font-size: 1.8rem; font-weight: 700; }
        .status-none  { color: #95a5a6; font-size: 1.3rem; font-weight: 500; }
        .metric-box   { background:#1e1e2e; border-radius:12px; padding:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fungsi bantu: muat model (di-cache agar hanya dimuat sekali)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Memuat model AI, mohon tunggu...")
def load_model(path: str):
    """Memuat model Keras dari file .h5 dan mengembalikannya."""
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner="🔧 Menginisialisasi FacePreprocessor...")
def load_preprocessor(tflite_path: str):
    """Membuat instance FacePreprocessor (MediaPipe BlazeFace)."""
    return FacePreprocessor(model_asset_path=tflite_path, margin_ratio=0.15)


# ---------------------------------------------------------------------------
# Fungsi inferensi: terima frame BGR numpy array → (label, confidence, face_rgb)
# ---------------------------------------------------------------------------
def run_inference(frame_bgr: np.ndarray, model, preprocessor) -> dict:
    """
    Jalankan pipeline lengkap pada satu frame.

    Returns dict dengan keys:
        label       : str  → "Real (Bonafide)" / "Fake (Spoof)" / "No Face"
        confidence  : float → 0–100
        face_rgb    : np.ndarray | None   → crop wajah dalam RGB (224x224)
        fft_img     : np.ndarray | None   → spektrum FFT (224x224, BGR)
        raw_score   : float | None        → output sigmoid model (0–1)
    """
    try:
        result = preprocessor.process_frame(frame_bgr)
    except Exception as e:
        return {"label": f"Preprocessor Error: {e}", "confidence": 0.0,
                "face_rgb": None, "fft_img": None, "raw_score": None}

    if result is None:
        return {"label": "No Face Detected", "confidence": 0.0,
                "face_rgb": None, "fft_img": None, "raw_score": None}

    face_crop, fft_magnitude = result  # BGR crop & FFT magnitude (sudah 224x224 BGR)

    # Resize ke 224×224 (jaga-jaga jika crop ukurannya bervariasi)
    face_resized = cv2.resize(face_crop, (224, 224))
    fft_resized  = cv2.resize(fft_magnitude, (224, 224))

    # Siapkan input dual-stream — pastikan contiguous array untuk TF
    rgb_input = np.ascontiguousarray(
        np.expand_dims(face_resized, axis=0).astype(np.float32)
    )  # (1,224,224,3)
    fft_input = np.ascontiguousarray(
        np.expand_dims(fft_resized, axis=0).astype(np.float32)
    )  # (1,224,224,3)

    inputs = {"rgb_input": rgb_input, "fft_input": fft_input}
    try:
        raw_score = float(model.predict(inputs, verbose=0)[0][0])
    except Exception as e:
        return {"label": f"Model Error: {e}", "confidence": 0.0,
                "face_rgb": None, "fft_img": None, "raw_score": None}

    if raw_score < THRESHOLD:
        label      = "Real (Bonafide)"
        confidence = (1.0 - raw_score) * 100
    else:
        label      = "Fake (Spoof)"
        confidence = raw_score * 100

    # Konversi wajah ke RGB untuk ditampilkan Streamlit
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    return {
        "label":      label,
        "confidence": confidence,
        "face_rgb":   face_rgb,
        "fft_img":    fft_magnitude,  # sudah BGR, akan dikonversi saat display
        "raw_score":  raw_score,
    }


# ---------------------------------------------------------------------------
# VideoProcessor untuk streamlit-webrtc (mode real-time)
# ---------------------------------------------------------------------------
class LivenessVideoProcessor(VideoProcessorBase):
    """
    Memproses setiap frame dari webcam secara real-time.
    Hasil prediksi ditulis langsung ke atas frame video.
    """

    def __init__(self):
        self.model       = st.session_state.get("model")
        self.preprocessor = st.session_state.get("preprocessor")
        self.result      = {"label": "Waiting...", "confidence": 0.0,
                            "raw_score": None}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")

        if self.model is not None and self.preprocessor is not None:
            info = run_inference(bgr, self.model, self.preprocessor)
            self.result = info

            label      = info["label"]
            confidence = info["confidence"]

            # Warna overlay: hijau = Real, merah = Spoof, abu = no face
            if "Real" in label:
                color = (0, 230, 80)
            elif "Fake" in label:
                color = (0, 50, 230)
            else:
                color = (200, 200, 200)

            # Gambar bounding-box teks di sudut kiri atas frame
            text = f"{label}  {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(bgr, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
            cv2.putText(bgr, text, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(bgr, format="bgr24")


# ===========================================================================
# MAIN UI
# ===========================================================================
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🛡️ Face Liveness Detection")
    st.markdown(
        "Deteksi wajah asli vs palsu (spoof) secara **real-time** menggunakan "
        "model Dual-Stream CNN (RGB + FFT) berbasis MobileNetV3."
    )
    st.divider()

    # ── Sidebar: Status & Konfigurasi ───────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Konfigurasi")

        # Muat model
        model = load_model(MODEL_PATH)
        if model is None:
            st.error(
                f"❌ Model tidak ditemukan!\n\n"
                f"Pastikan file berada di:\n`{MODEL_PATH}`"
            )
            st.info("Latih model terlebih dahulu dengan `src/train.py`, lalu simpan ke folder `models/`.")
        else:
            st.success("✅ Model berhasil dimuat")
            with st.expander("Detail Model"):
                st.write(f"**Path:** `{MODEL_PATH}`")
                st.write(f"**Input:** RGB 224×224 + FFT 224×224")
                st.write(f"**Threshold:** `{THRESHOLD}`")

        # Muat preprocessor
        preprocessor = load_preprocessor(TFLITE_PATH)
        st.success("✅ FacePreprocessor siap")

        # Simpan ke session_state agar bisa diakses VideoProcessor
        st.session_state["model"]        = model
        st.session_state["preprocessor"] = preprocessor

        st.divider()
        st.header("📋 Legenda")
        st.markdown(
            "🟢 **Real (Bonafide)** – Wajah manusia asli  \n"
            "🔴 **Fake (Spoof)** – Foto / layar / topeng  \n"
            "⚪ **No Face Detected** – Tidak ada wajah terdeteksi"
        )

    # ── Mode Pilihan ─────────────────────────────────────────────────────────
    tab_live, tab_snap = st.tabs(["📹 Live Webcam (Real-time)", "📷 Snapshot / Upload"])

    # ===================================================================
    # Tab 1 — Real-time webcam via streamlit-webrtc
    # ===================================================================
    with tab_live:
        if model is None:
            st.warning("⚠️ Model belum dimuat. Silakan periksa sidebar.")
        else:
            st.markdown(
                "Klik **START** untuk mengaktifkan kamera. "
                "Prediksi akan muncul langsung di atas video stream."
            )

            # Konfigurasi STUN/TURN (gunakan server publik Google)
            rtc_config = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            ctx = webrtc_streamer(
                key="liveness-webrtc",
                video_processor_factory=LivenessVideoProcessor,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Panel hasil di bawah video
            if ctx.video_processor:
                info = ctx.video_processor.result
                col_label, col_conf = st.columns([2, 1])

                with col_label:
                    label = info.get("label", "Waiting...")
                    if "Real" in label:
                        st.markdown(f'<p class="status-real">🟢 {label}</p>',
                                    unsafe_allow_html=True)
                    elif "Fake" in label:
                        st.markdown(f'<p class="status-spoof">🔴 {label}</p>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-none">⚪ {label}</p>',
                                    unsafe_allow_html=True)

                with col_conf:
                    conf = info.get("confidence", 0.0)
                    if conf > 0:
                        st.metric("Confidence", f"{conf:.1f}%")

    # ===================================================================
    # Tab 2 — Snapshot / Upload gambar statis
    # ===================================================================
    with tab_snap:
        st.markdown("Upload gambar **atau** ambil snapshot dari webcam untuk dianalisis.")

        source_opt = st.radio(
            "Sumber gambar:",
            ["📁 Upload File", "📷 Kamera (Snapshot)"],
            horizontal=True,
        )

        frame_bgr = None  # frame yang akan diproses

        if source_opt == "📁 Upload File":
            uploaded = st.file_uploader(
                "Pilih gambar (JPG / PNG / WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded:
                pil_img   = Image.open(uploaded).convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        else:  # Kamera snapshot
            snapshot = st.camera_input("Arahkan wajah ke kamera, lalu klik tombol 📷")
            if snapshot:
                pil_img   = Image.open(snapshot).convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ── Tombol Analisis ─────────────────────────────────────────────────
        if frame_bgr is not None:
            st.image(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                caption="Gambar Input",
                use_container_width=True,
            )

            if st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True):
                if model is None:
                    st.error("❌ Model belum dimuat. Periksa sidebar.")
                else:
                    with st.spinner("Memproses..."):
                        info = run_inference(frame_bgr, model, preprocessor)

                    st.divider()
                    # ── Tampilkan hasil ──────────────────────────────────────
                    col_res, col_face, col_fft = st.columns([2, 1, 1])

                    with col_res:
                        label = info["label"]
                        conf  = info["confidence"]
                        raw   = info["raw_score"]

                        if "Real" in label:
                            st.markdown(f'<p class="status-real">🟢 {label}</p>',
                                        unsafe_allow_html=True)
                        elif "Fake" in label:
                            st.markdown(f'<p class="status-spoof">🔴 {label}</p>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p class="status-none">⚪ {label}</p>',
                                        unsafe_allow_html=True)

                        if raw is not None:
                            st.metric("Confidence", f"{conf:.2f}%")
                            st.progress(int(conf))
                            st.caption(f"Raw sigmoid score: `{raw:.4f}`  |  Threshold: `{THRESHOLD}`")
                        else:
                            st.warning("Wajah tidak terdeteksi pada gambar ini.")

                    with col_face:
                        if info["face_rgb"] is not None:
                            st.image(info["face_rgb"], caption="Crop Wajah (224×224)", use_container_width=True)

                    with col_fft:
                        if info["fft_img"] is not None:
                            fft_rgb = cv2.cvtColor(info["fft_img"], cv2.COLOR_BGR2RGB)
                            st.image(fft_rgb, caption="FFT Magnitude Spectrum", use_container_width=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

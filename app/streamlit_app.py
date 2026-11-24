"""streamlit app - Web User Interface for Face Anti-Spoofing (Liveness Detection).
Jalankan dari root project::
    streamlit run app/streamlit_app.py

Alur data:
Webcam (WebRTC thread) -> FacePreprocessor (MediaPipe + 2D FFT)
-> model MobileNetV3 hasil training -> status prediksi real-time.

Catatan desain:
* Model dimuat sekali lalu di-cache oleh Streamlit (`st.cache_resource`).
* Inferensi berjalan di thread streamlit-webrtc, bukan di thread UI.
* Hasil inferensi dibagikan ke UI lewat `LiveState` yang thread-safe.
* Status panel di-refresh otomatis, sedangkan overlay pada video real-time.
* UI memakai kartu membulat, aksen hijau, dan tipografi Plus Jakarta Sans.
"""
from __future__ import annotations
import html
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import streamlit as st
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Path & konfigurasi project
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_paths import discover_model_files as _discover_model_files
from src.model_paths import resolve_default_model_path

MODELS_DIR = ROOT / "models"
FACE_MODEL_PATH = ROOT / "blaze_face_short_range.tflite"
# Otomatis memilih *.keras terbaru bila ada, lalu model.weights.h5, lalu *.h5 lain.
DEFAULT_MODEL_PATH = resolve_default_model_path(MODELS_DIR)
DEFAULT_INPUT_SIZE = (224, 224)
# File model yang didukung (full model .keras/.h5 atau bobot .weights.h5)
SUPPORTED_MODEL_EXTENSIONS = ("*.keras", "*.h5", "*.hdf5")

# File .h5 kosong/terpotong biasanya hanya berisi header HDF5 (~1 KB).
MIN_VALID_MODEL_BYTES = 1024



# Streamlit-WebRTC bersifat opsional: jika gagal di-import, UI otomatis
# menawarkan mode snapshot OpenCV agar aplikasi tetap bisa didemokan.
try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoHTMLAttributes
    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - tergantung environment
    av = None  # type: ignore[assignment]
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    class VideoProcessorBase:  # type: ignore[no-redef]
        """Fallback dummy base class ketika streamlit-webrtc tidak tersedia."""
        pass

    class VideoHTMLAttributes(dict):  # type: ignore[no-redef]
        pass

    def webrtc_streamer(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError(f"streamlit-webrtc tidak tersedia: {WEBRTC_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Hasil satu kali inferensi pada satu frame."""
    ok: bool
    spoof_score: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    face_rgb: np.ndarray | None = None
    fft_rgb: np.ndarray | None = None
    message: str = ""


@dataclass
class LiveSnapshot:
    """Snapshot status yang aman dibaca oleh thread UI."""
    status: str = "WAITING"  # WAITING | REAL | SPOOF | NO_FACE | ERROR
    label: str = "Waiting for camera"
    spoof_score: float = 0.0
    confidence: float = 0.0
    fps: float = 0.0
    frame_index: int = 0
    bbox: tuple[int, int, int, int] | None = None
    message: str = "Press START to enable the camera."
    face_rgb: np.ndarray | None = None
    fft_rgb: np.ndarray | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class RuntimeSettings:
    """Pengaturan yang bisa diubah user tanpa me-restart stream WebRTC."""
    threshold: float = 0.5
    smoothing: int = 7
    mirror: bool = True
    predict_interval: float = 0.20
    show_previews: bool = False


class LiveState:
    """State bersama antara worker WebRTC dan script thread Streamlit."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = LiveSnapshot()
        self._history: deque[tuple[float, float]] = deque(maxlen=180)

    def publish(self, snapshot: LiveSnapshot, score: float | None = None) -> None:
        with self._lock:
            self._snapshot = snapshot
            if score is not None:
                self._history.append((time.time(), float(score)))

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            # Salin referensi: array numpy tidak diubah setelah dipublikasikan.
            return replace(self._snapshot)

    def history(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            self._snapshot = LiveSnapshot()
            self._history.clear()


# ---------------------------------------------------------------------------
# Engine inferensi (frame -> FacePreprocessor -> model)
# ---------------------------------------------------------------------------
class LivenessEngine:
    """Membungkus FacePreprocessor + model Keras dari Role 2."""
    def __init__(
        self,
        model_path: str | Path,
        face_model_path: str | Path = FACE_MODEL_PATH,
        margin_ratio: float = 0.15,
    ) -> None:
        # Import lambat supaya UI tetap bisa tampil walaupun dependency bermasalah.
        import tensorflow as tf
        from src.preprocessing import FacePreprocessor

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        if not Path(face_model_path).exists():
            raise FileNotFoundError(
                f"Face detector not found: {face_model_path}. "
                "Make sure blaze_face_short_range.tflite exists in the project root."
            )

        self.model = self._load_keras_model(self.model_path)
        self.input_names = self._resolve_input_names()
        self.num_inputs = len(self.model.inputs)
        self._sizes = [self._input_size(inp) for inp in self.model.inputs]
        self._fft_index = self._resolve_fft_index()

        self.preprocessor = FacePreprocessor(
            model_asset_path=str(face_model_path),
            margin_ratio=margin_ratio,
        )

        # Kompilasi inferensi ke TensorFlow Graph (tf.function) agar performa inferensi 15-20x lebih cepat
        @tf.function(reduce_retracing=True)
        def _predict_step(feed_dict: dict[str, tf.Tensor]) -> tf.Tensor:
            return self.model(feed_dict, training=False)

        self._predict_step = _predict_step
        self._warmup()

    @staticmethod
    def _load_keras_model(model_path: Path):
        import json
        import tensorflow as tf

        # File *.weights.h5 hanya berisi bobot. Jangan coba load_model()
        # karena pasti gagal dan memakan waktu; langsung susun dari config.
        weights_only = model_path.name.endswith((".weights.h5", ".weights.hdf5"))
        load_err: Exception | None = None
        if not weights_only:
            try:
                return tf.keras.models.load_model(str(model_path), compile=False)
            except Exception as err:
                load_err = err

        config_candidates = [
            model_path.parent / "config.json",
            model_path.parent / f"{model_path.stem}.config.json",
            MODELS_DIR / "config.json",
        ]
        for config_path in config_candidates:
            if not config_path.exists():
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                model = tf.keras.models.model_from_json(json.dumps(cfg))
                model.load_weights(str(model_path))
                return model
            except Exception as err:
                load_err = err
                continue

        raise load_err or RuntimeError(f"Model tidak dapat dimuat: {model_path}")

    # -- model introspection -------------------------------------------------
    def _resolve_input_names(self) -> list[str]:
        try:
            names = [str(name).split(":")[0] for name in self.model.input_names]
        except Exception:
            names = []
        if len(names) != len(self.model.inputs):
            names = [str(inp.name).split(":")[0] for inp in self.model.inputs]
        return names

    @staticmethod
    def _input_size(tensor: Any) -> tuple[int, int]:
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 4:
            height, width = shape[1], shape[2]
            if height and width:
                return int(width), int(height)
        return DEFAULT_INPUT_SIZE

    def _resolve_fft_index(self) -> int | None:
        if self.num_inputs < 2:
            return None
        keywords = ("fft", "freq", "spectrum", "spectral")
        for index, name in enumerate(self.input_names):
            if any(keyword in name.lower() for keyword in keywords):
                return index
        # Konvensi project: input kedua adalah FFT.
        return 1

    @property
    def rgb_size(self) -> tuple[int, int]:
        if self.num_inputs == 1:
            return self._sizes[0]
        rgb_index = 0 if self._fft_index != 0 else 1
        return self._sizes[rgb_index]

    @property
    def fft_size(self) -> tuple[int, int]:
        if self._fft_index is None:
            return self.rgb_size
        return self._sizes[self._fft_index]

    def _warmup(self) -> None:
        """Satu inferensi dummy agar frame pertama tidak terasa lambat."""
        rgb = np.zeros((*self.rgb_size[::-1], 3), dtype=np.float32)
        fft = np.zeros((*self.fft_size[::-1], 3), dtype=np.float32)
        try:
            self._run_model(rgb, fft)
        except Exception:
            # Warmup bersifat opsional; jangan gagalkan pemuatan model.
            pass

    # -- inference -----------------------------------------------------------
    def _run_model(self, rgb: np.ndarray, fft: np.ndarray) -> float:
        import tensorflow as tf

        if self.num_inputs >= 2:
            feed: dict[str, tf.Tensor] = {}
            for index, name in enumerate(self.input_names):
                image = fft if index == self._fft_index else rgb
                feed[name] = tf.convert_to_tensor(np.expand_dims(image.astype(np.float32), axis=0))
        else:
            feed = {self.input_names[0]: tf.convert_to_tensor(np.expand_dims(rgb.astype(np.float32), axis=0))}

        try:
            output = self._predict_step(feed)
        except Exception:
            output = self.model(feed, training=False)

        score = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(score):
            raise ValueError("Model menghasilkan skor yang tidak valid (NaN/Inf).")
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Jalankan pipeline lengkap untuk satu frame BGR."""
        processed = self.preprocessor.process_frame(frame_bgr, return_bbox=True)
        if processed is None:
            return DetectionResult(ok=False, message="No face detected")

        face_bgr, fft_bgr, bbox = processed
        rgb = cv2.resize(face_bgr, self.rgb_size)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        fft = cv2.resize(fft_bgr, self.fft_size)
        # FFT grayscale: konversi ini hanya menyamakan urutan channel dengan RGB.
        fft = cv2.cvtColor(fft, cv2.COLOR_BGR2RGB)

        score = self._run_model(rgb, fft)
        return DetectionResult(
            ok=True,
            spoof_score=score,
            bbox=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
            face_rgb=rgb,
            fft_rgb=fft,
            message="OK",
        )

    # -- info untuk tab Model ------------------------------------------------
    def describe(self) -> dict[str, Any]:
        inputs = []
        for name, tensor in zip(self.input_names, self.model.inputs):
            shape = ", ".join(str(dim) for dim in tensor.shape)
            inputs.append({"name": name, "shape": shape})

        outputs = []
        for tensor in self.model.outputs:
            shape = ", ".join(str(dim) for dim in tensor.shape)
            outputs.append({"shape": shape})

        return {
            "path": str(self.model_path),
            "inputs": inputs,
            "outputs": outputs,
            "params": int(self.model.count_params()),
            "stream": "dual-stream (RGB + FFT)" if self.num_inputs >= 2 else "single-stream (RGB)",
        }


@st.cache_resource(show_spinner=False)
def load_engine(model_path: str, face_model_path: str, margin_ratio: float) -> LivenessEngine:
    """Cache model agar tidak dimuat ulang di setiap rerun Streamlit."""
    return LivenessEngine(model_path, face_model_path, margin_ratio)


# ---------------------------------------------------------------------------
# Snapshot helpers & overlay
# ---------------------------------------------------------------------------
def build_snapshot(
    result: DetectionResult,
    score: float,
    threshold: float,
    fps: float,
    frame_index: int,
    window: int = 1,
) -> LiveSnapshot:
    """Ubah hasil inferensi (+ skor yang sudah dihaluskan) menjadi snapshot UI."""
    if not result.ok:
        no_face = "no face detected" in result.message.lower()
        return LiveSnapshot(
            status="NO_FACE" if no_face else "ERROR",
            label="No face detected" if no_face else "Processing error",
            fps=round(fps, 1),
            frame_index=frame_index,
            message=result.message,
            face_rgb=result.face_rgb,
            fft_rgb=result.fft_rgb,
        )

    is_spoof = score >= threshold
    confidence = score * 100.0 if is_spoof else (1.0 - score) * 100.0
    label = "PALSU (SPOOF)" if is_spoof else "ASLI (BONAFIDE)"
    window_note = f"avg over {max(1, window)} frames" if window > 1 else "this frame"

    return LiveSnapshot(
        status="SPOOF" if is_spoof else "REAL",
        label=label,
        spoof_score=float(score),
        confidence=float(confidence),
        fps=round(fps, 1),
        frame_index=frame_index,
        bbox=result.bbox,
        message=f"Spoof score {score * 100:.1f}% | {window_note} | threshold {threshold * 100:.0f}%",
        face_rgb=result.face_rgb,
        fft_rgb=result.fft_rgb,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    snapshot: LiveSnapshot,
    fps: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Draw a thin terracotta bounding box and a compact label chip."""
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image

    def ascii_only(text: str) -> str:
        # cv2.putText only supports ASCII characters.
        return text.encode("ascii", "ignore").decode("ascii")

    terracotta = (59, 97, 200)   # #c8613b in BGR
    charcoal = (43, 36, 32)      # #20242b in BGR
    white = (255, 255, 255)

    def chip(
        text: str,
        origin_x: int,
        origin_y: int,
        scale: float = 0.45,
        text_color: tuple[int, int, int] = charcoal,
    ) -> None:
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        pad_x, pad_y = 9, 6
        x1 = max(0, min(origin_x, width - text_w - pad_x * 2 - 2))
        y1 = max(0, min(origin_y, height - text_h - pad_y * 2 - 2))
        x2, y2 = x1 + text_w + pad_x * 2, y1 + text_h + pad_y * 2

        cv2.rectangle(image, (x1, y1), (x2, y2), white, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), terracotta, 1, cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (x1 + pad_x, y1 + pad_y + text_h),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    if snapshot.bbox is not None:
        x1, y1, x2, y2 = snapshot.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(image, (x1, y1), (x2, y2), terracotta, 2, cv2.LINE_AA)
            label = ascii_only(f"{snapshot.label.upper()}  {snapshot.confidence:.0f}%")
            (chip_w, chip_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            chip(label, x1, max(0, y1 - chip_h - 16))

    # Status chip di pojok kiri atas
    status = ascii_only(snapshot.label.upper())
    chip(status, 16, 16)

    # FPS badge di pojok kanan atas agar selalu tampak jelas dan tidak terpotong
    fps_val = max(0.0, fps)
    fps_text = ascii_only(f"{fps_val:.1f} FPS")
    (fps_w, fps_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)
    pad_x, pad_y = 8, 5
    x1 = max(0, width - fps_w - pad_x * 2 - 16)
    y1 = 16
    x2, y2 = x1 + fps_w + pad_x * 2, y1 + fps_h + pad_y * 2

    # Gambar kotak background putih dengan border terracotta
    cv2.rectangle(image, (x1, y1), (x2, y2), white, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), terracotta, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        fps_text,
        (x1 + pad_x, y1 + pad_y + fps_h),
        cv2.FONT_HERSHEY_DUPLEX,
        0.40,
        charcoal,
        1,
        cv2.LINE_AA,
    )

    return image


# ---------------------------------------------------------------------------
# WebRTC video processor
# ---------------------------------------------------------------------------
class LivenessVideoProcessor(VideoProcessorBase):
    """Dipanggil streamlit-webrtc untuk setiap frame kamera."""
    def __init__(self, engine: LivenessEngine, settings: RuntimeSettings, state: LiveState):
        self.engine = engine
        self.settings = settings
        self.state = state
        self._scores: deque[float] = deque(maxlen=60)
        self._frame_index = 0
        self._last_frame_at = time.perf_counter()
        self._last_predict_at = 0.0
        self._fps = 0.0
        self._snapshot = LiveSnapshot(message="Menunggu frame pertama...")

        # Worker thread inferensi asinkron agar tidak memblokir render frame video
        self._pending_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()

    def _inference_loop(self) -> None:
        """Loop latar belakang khusus inferensi model agar loop video (recv) tetap ~30 FPS."""
        while not self._stop_event.is_set():
            frame_to_process = None
            with self._frame_lock:
                if self._pending_frame is not None:
                    frame_to_process = self._pending_frame
                    self._pending_frame = None

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            try:
                result = self.engine.predict(frame_to_process)
            except Exception as exc:
                self._scores.clear()
                self._snapshot = LiveSnapshot(
                    status="ERROR",
                    label="Terjadi kesalahan",
                    fps=round(self._fps, 1),
                    frame_index=self._frame_index,
                    message=f"Error inferensi: {exc}",
                )
                self.state.publish(self._snapshot)
            else:
                if result.ok:
                    self._scores.append(result.spoof_score)
                    window = max(1, int(self.settings.smoothing))
                    recent = list(self._scores)[-window:]
                    score = float(np.mean(recent))
                    self._snapshot = build_snapshot(
                        result,
                        score,
                        self.settings.threshold,
                        self._fps,
                        self._frame_index,
                        window=len(recent),
                    )
                    self.state.publish(self._snapshot, score=score)
                else:
                    self._scores.clear()
                    self._snapshot = build_snapshot(
                        result,
                        0.0,
                        self.settings.threshold,
                        self._fps,
                        self._frame_index,
                    )
                    self.state.publish(self._snapshot)

            # Jeda antar inferensi sesuai predict_interval (default 0.15 - 0.20s)
            time.sleep(self.settings.predict_interval)

    def _update_fps(self) -> None:
        now = time.perf_counter()
        delta = now - self._last_frame_at
        self._last_frame_at = now
        if delta <= 0:
            return
        instant = 1.0 / delta
        self._fps = instant if self._fps == 0 else (0.85 * self._fps + 0.15 * instant)

    def recv(self, frame: Any) -> Any:
        image = frame.to_ndarray(format="bgr24")
        if self.settings.mirror:
            image = cv2.flip(image, 1)

        self._frame_index += 1
        self._update_fps()

        # Kirim frame ke worker hanya jika worker sedang idle (tidak menumpuk copy frame)
        if self._pending_frame is None and self._frame_lock.acquire(blocking=False):
            try:
                self._pending_frame = image.copy()
            finally:
                self._frame_lock.release()

        annotated = draw_overlay(
            image,
            self._snapshot,
            self._fps,
            threshold=self.settings.threshold,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def __del__(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# UI helpers & STREAMLIT APP STYLING
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    try:
        import yaml
        with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def discover_models() -> list[Path]:
    return _discover_model_files(MODELS_DIR)


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_live_state() -> LiveState:
    if "live_state" not in st.session_state:
        st.session_state.live_state = LiveState()
    return st.session_state.live_state


def get_runtime_settings(
    threshold: float,
    smoothing: int,
    mirror: bool,
    show_previews: bool,
) -> RuntimeSettings:
    """Objek setting yang persisten agar worker WebRTC melihat perubahan UI."""
    settings = st.session_state.get("runtime_settings")
    if not isinstance(settings, RuntimeSettings):
        settings = RuntimeSettings()
    st.session_state.runtime_settings = settings

    settings.threshold = float(threshold)
    settings.smoothing = int(smoothing)
    settings.mirror = bool(mirror)
    settings.show_previews = bool(show_previews)
    return settings


def render_html(markup: str) -> None:
    """Render HTML/CSS tanpa memicu code block Markdown akibat indentasi."""
    cleaned = "\n".join(line.lstrip() for line in markup.strip().splitlines())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STYLE APLIKASI (kartu membulat, aksen hijau, tipografi kuat)
# ---------------------------------------------------------------------------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --page: #F3F5FC;
    --surface: #E5E9FB;
    --strong: #CED5F6;
    --brand: #184093;
    --brand-hover: #3D60A8;
    --secondary: #4664AA;
    --muted: #788FC3;
    --cta: #00A806;
    --cta-hover: #90E293;
    --cta-ink: #011A02;
    --hl: #FDCA4B;
    --hl-ink: #3C2E08;
    --danger: #CF3B2B;
    --danger-soft: #FDECEB;
    --white: #FFFFFF;
    --radius: 26px;
}

html, body, .stApp {
    background: var(--page) !important;
    color: var(--brand) !important;
    font-family: 'Space Grotesk', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp :is(h1, h2, h3, h4, h5, h6) {
    font-family: 'Plus Jakarta Sans', 'Helvetica Neue', Arial, sans-serif !important;
}
[data-testid="stHeader"] { background: transparent; height: 0; pointer-events: none; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer {
    display: none !important;
}
/* Toolbar tetap ada agar tombol buka/tutup sidebar tidak ikut hilang. */
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
[data-testid="stExpandSidebarButton"] {
    position: fixed !important;
    top: 1rem !important;
    left: 2.05rem !important;
    z-index: 1000000 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    padding: 0 !important;
    border: 1px solid var(--strong) !important;
    border-radius: 999px !important;
    background: var(--white) !important;
    color: var(--brand) !important;
    visibility: visible !important;
    pointer-events: auto !important;
    box-shadow: 0 2px 8px rgba(9, 31, 80, 0.14) !important;
    transition: background 0.2s ease;
}
[data-testid="stExpandSidebarButton"]:hover { background: var(--surface) !important; }
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    color: var(--brand) !important; font-size: 19px !important;
}
.stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .app-nav {
    padding-left: 3.15rem;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] button {
    width: 28px !important; height: 28px !important;
    border: 1px solid var(--strong) !important; border-radius: 999px !important;
    background: var(--white) !important;
    box-shadow: 0 2px 8px rgba(9, 31, 80, 0.10);
}
[data-testid="stSidebarCollapseButton"] button:hover { background: var(--surface) !important; }
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    color: var(--brand) !important; font-size: 18px !important;
}
[data-testid="stMain"] { background: var(--page); }
.block-container {
    max-width: 1440px;
    padding: 0.6rem 1.6rem 0.7rem !important;
}

/* ---------- navbar pil ---------- */
.app-nav {
    display: flex; align-items: center; gap: 1rem;
    background: rgba(255, 255, 255, 0.97);
    border-radius: var(--radius);
    padding: 0.5rem 0.7rem 0.5rem 0.85rem;
    box-shadow: 0 6px 20px rgba(9, 31, 80, 0.13);
    margin-bottom: 0.85rem;
}
.app-brand { display: flex; align-items: center; gap: 0.55rem; min-width: 0; }
.app-logo {
    width: 30px; height: 30px; border-radius: 10px; flex: 0 0 auto;
    display: grid; place-items: center;
    background: linear-gradient(135deg, var(--brand) 10%, var(--cta) 130%);
    color: var(--white);
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 0.9rem;
}
.app-brand-name {
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
    font-size: 1rem; letter-spacing: -0.025em; color: var(--brand); white-space: nowrap;
}
.app-brand-sub {
    color: var(--secondary); font-size: 0.75rem; font-weight: 500;
    padding-left: 0.6rem; border-left: 1px solid var(--strong); white-space: nowrap;
}
.app-nav-meta { margin-left: auto; display: flex; align-items: center; gap: 0.35rem; }
.app-chip {
    display: inline-flex; align-items: center; gap: 0.32rem;
    background: var(--surface); color: var(--brand);
    border-radius: 999px; padding: 0.28rem 0.7rem;
    font-size: 0.7rem; font-weight: 600; white-space: nowrap;
}
.app-chip--live { background: #DFF6DF; color: #0B6B0C; }
.app-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto; background: var(--cta); }

/* ---------- hero ---------- */
.app-hero {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 1.4rem; margin: 0.15rem 0 0.9rem;
}
.app-tag {
    display: inline-block; border: 1px solid var(--strong); border-radius: 999px;
    padding: 0.16rem 0.65rem; background: var(--white);
    color: var(--secondary); font-size: 0.66rem; font-weight: 600;
    letter-spacing: 0.01em;
}
.app-hero h1 {
    margin: 0.32rem 0 0;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: clamp(1.5rem, 2.15vw, 2rem);
    font-weight: 800; line-height: 1.06; letter-spacing: -0.03em;
    color: var(--brand);
}
.app-mark {
    background: linear-gradient(transparent 60%, var(--hl) 60%);
    padding: 0 0.12em;
}
.app-grad {
    background: linear-gradient(105deg, var(--brand) 20%, var(--cta) 105%);
    -webkit-background-clip: text; background-clip: text;
    color: transparent;
}
.app-lead {
    max-width: 345px; margin: 0 0 0.15rem;
    font-size: 0.8rem; line-height: 1.5; color: var(--secondary); font-weight: 500;
}

/* ---------- panel kamera ---------- */
.st-key-app_viewport {
    background: var(--white) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem 0.9rem 0.9rem !important;
    box-shadow: 0 16px 40px rgba(24, 64, 147, 0.10) !important;
}
.st-key-app_viewport [data-testid="stVerticalBlock"] { gap: 0.45rem; }
.st-key-app_viewport iframe {
    width: 100% !important;
    height: min(46vh, 410px) !important;
    border: 0 !important;
    border-radius: 18px !important;
    background: var(--surface) !important;
}
.st-key-app_viewport [data-testid="stIFrame"] {
    height: min(46vh, 410px) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
}
.app-vp-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.8rem; padding: 0.05rem 0.15rem 0;
}
.app-mono {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.63rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
}
.app-state {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--surface); color: var(--brand);
    border-radius: 999px; padding: 0.25rem 0.62rem;
    font-size: 0.68rem; font-weight: 600; white-space: nowrap;
}
.app-state-sep { width: 1px; height: 10px; background: var(--strong); margin: 0 0.1rem; }

/* ---------- panel hasil ---------- */
.app-panel {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1rem 1.1rem 0.9rem;
    box-shadow: 0 2px 0 rgba(255, 255, 255, 0.65) inset;
}
.app-panel-top { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; }
.app-panel-top .app-chip { background: var(--white); }
.app-verdict {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 0.5rem 0 0.2rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(1.2rem, 1.55vw, 1.5rem);
    font-weight: 800; letter-spacing: -0.025em; line-height: 1.1;
}
.app-verdict-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.app-note { margin: 0 0 0.65rem; font-size: 0.75rem; line-height: 1.4; color: var(--secondary); font-weight: 500; }
.app-meter { height: 8px; border-radius: 999px; background: var(--white); overflow: hidden; }
.app-meter > span { display: block; height: 100%; border-radius: 999px; transition: width 0.3s ease; }
.app-meter-row {
    display: flex; justify-content: space-between; margin-top: 0.28rem;
    font-size: 0.66rem; font-weight: 600; color: var(--secondary);
}
.app-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; margin-top: 0.7rem; }
.app-stat { background: var(--white); border-radius: 15px; padding: 0.45rem 0.7rem 0.5rem; }
.app-stat span {
    display: block; font-size: 0.58rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
}
.app-stat strong {
    display: block; margin-top: 0.14rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.02rem; font-weight: 800; letter-spacing: -0.02em; color: var(--brand);
    font-variant-numeric: tabular-nums;
}
.app-panel-foot {
    display: flex; flex-wrap: wrap; gap: 0.25rem 0.7rem;
    margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid var(--strong);
    font-size: 0.63rem; font-weight: 500; color: var(--secondary);
}

/* ---------- tabs sebagai segmented pill ---------- */
.stTabs { margin-top: 0 !important; }
.stTabs [role="tablist"] {
    display: flex; gap: 0.2rem; width: max-content; background: var(--surface);
    border: none; border-radius: 999px; padding: 0.22rem;
    margin-bottom: 0.6rem;
}
.stTabs [data-testid="stTab"] {
    display: flex; align-items: center; height: auto;
    padding: 0.32rem 0.95rem; border-radius: 999px;
    background: transparent; color: var(--secondary);
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.78rem;
    letter-spacing: -0.01em; cursor: pointer;
}
.stTabs [data-testid="stTab"]:hover { background: rgba(255, 255, 255, 0.55); color: var(--brand); }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    background: var(--white) !important; color: var(--brand) !important;
    box-shadow: 0 2px 8px rgba(10, 31, 80, 0.14);
}
.stTabs [data-testid="stTab"] p { color: inherit !important; font-weight: 700 !important; }
.stTabs .react-aria-SelectionIndicator { display: none !important; }
.stTabs [data-testid="stTabPanel"] { padding: 0 !important; }
.stTabs [data-testid="stVerticalBlock"] { gap: 0.45rem; }

/* ---------- tombol ---------- */
.stButton > button, .stDownloadButton > button {
    background: var(--cta); color: var(--cta-ink);
    border: none; border-radius: 999px;
    padding: 0.46rem 1.1rem;
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 0.8rem;
    letter-spacing: 0.01em;
    box-shadow: 0 6px 14px rgba(0, 168, 6, 0.22);
    transition: background 0.2s ease, transform 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--cta-hover); color: var(--cta-ink); border: none;
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus:not(:active) { border: none; color: var(--cta-ink); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--page); border-right: none;
}
[data-testid="stSidebar"] > div:first-child { background: var(--page); }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
[data-testid="stSidebar"] h2 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.02rem; font-weight: 800; letter-spacing: -0.02em; color: var(--brand);
    padding-bottom: 0.55rem; border-bottom: 1px solid var(--strong);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.76rem; font-weight: 600; color: var(--brand);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] *,
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button { width: 100%; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.68rem; color: var(--muted);
}
[data-testid="stSidebar"] [data-testid="stAlert"] { border-radius: 16px; }

/* ---------- tab model ---------- */
.app-model-head { margin-bottom: 0.1rem; }
.app-model-head h2 {
    margin: 0.15rem 0 0.2rem;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.25rem; font-weight: 800; letter-spacing: -0.025em; color: var(--brand);
}
.app-model-head p { margin: 0; color: var(--secondary); font-size: 0.78rem; }
[data-testid="stMetric"] {
    background: var(--white); border-radius: 18px;
    padding: 0.6rem 0.85rem 0.65rem;
    box-shadow: 0 6px 18px rgba(24, 64, 147, 0.07);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.6rem !important; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
}
[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem; font-weight: 800; color: var(--brand); letter-spacing: -0.02em;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 16px; overflow: hidden; background: var(--white);
}
[data-testid="stExpander"] details { border-radius: 16px; border-color: var(--strong); background: var(--white); }

/* ---------- satu layar ---------- */
@media (min-width: 1100px) and (min-height: 640px) {
    [data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
    [data-testid="stHorizontalBlock"] { gap: 1rem !important; align-items: flex-start; }
}

@media (max-width: 1100px) {
    .app-hero { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    .app-lead { max-width: none; }
    .app-brand-sub { display: none; }
    .st-key-app_viewport iframe, .st-key-app_viewport [data-testid="stIFrame"] { height: 320px !important; }
}
</style>
"""


def inject_css() -> None:
    render_html(APP_CSS)


def render_header() -> None:
    render_html(
        """
        <div class="app-nav">
            <div class="app-brand">
                <span class="app-logo">s</span>
                <span class="app-brand-name">streamlit app</span>
                <span class="app-brand-sub">face anti-spoofing</span>
            </div>
            <div class="app-nav-meta">
                <span class="app-chip">MobileNetV3</span>
                <span class="app-chip app-chip--live"><span class="app-dot"></span> WebRTC</span>
            </div>
        </div>
        <div class="app-hero">
            <div>
                <span class="app-tag">Computer vision &middot; CBT</span>
                <h1>Wajah <span class="app-mark">asli</span>, <span class="app-grad">Anti Spoofing</span></h1>
            </div>
            <p class="app-lead">
                Verifikasi liveness dari webcam. Model memisahkan wajah
                bonafide dari serangan spoof seperti foto, layar, atau cetakan.
            </p>
        </div>
        """
    )


STATUS_LABEL = {
    "REAL": "Wajah asli",
    "SPOOF": "Spoof terdeteksi",
    "NO_FACE": "Tidak ada wajah",
    "ERROR": "Terjadi kesalahan",
    "WAITING": "Menunggu kamera",
}

STATUS_DOT = {
    "REAL": "#00A806",
    "SPOOF": "#CF3B2B",
    "NO_FACE": "#FDCA4B",
    "ERROR": "#CF3B2B",
    "WAITING": "#788FC3",
}

STATUS_TONE = {
    "REAL": "#184093",
    "SPOOF": "#CF3B2B",
    "NO_FACE": "#184093",
    "ERROR": "#CF3B2B",
    "WAITING": "#4664AA",
}


def _fallback_snapshot() -> LiveSnapshot:
    return LiveSnapshot(
        status="WAITING",
        label="Waiting for camera",
        message="Tekan START pada panel kamera untuk memulai deteksi.",
    )


def render_viewport_header(snapshot: LiveSnapshot | None, threshold: float) -> None:
    status = (snapshot or _fallback_snapshot()).status
    text = STATUS_LABEL.get(status, "Menunggu kamera")
    dot = STATUS_DOT.get(status, STATUS_DOT["WAITING"])
    render_html(
        f"""
        <div class="app-vp-head">
            <span class="app-mono">Live camera</span>
            <span class="app-state">
                <span class="app-dot" style="background: {dot};"></span>
                {text}
                <span class="app-state-sep"></span>
                {threshold * 100:.0f}%
            </span>
        </div>
        """
    )


def render_side_panel(snapshot: LiveSnapshot | None, threshold: float, active: bool) -> None:
    if snapshot is None or not active:
        snapshot = _fallback_snapshot()

    status = snapshot.status
    status_text = STATUS_LABEL.get(status, "Menunggu kamera")
    dot = STATUS_DOT.get(status, STATUS_DOT["WAITING"])
    tone = STATUS_TONE.get(status, STATUS_TONE["WAITING"])
    waiting = status == "WAITING"

    confidence = "--" if waiting else f"{snapshot.confidence:.1f}%"
    spoof = "--" if waiting else f"{snapshot.spoof_score * 100:.1f}%"
    width = 0.0 if waiting else min(max(snapshot.spoof_score, 0.0), 1.0) * 100
    bar = "#CF3B2B" if status == "SPOOF" else "#00A806"
    if not waiting and status != "SPOOF" and snapshot.spoof_score >= threshold * 0.75:
        bar = "#FDCA4B"

    render_html(
        f"""
        <aside class="app-panel">
            <div class="app-panel-top">
                <span class="app-mono">Hasil live</span>
                <span class="app-chip">Ambang {threshold * 100:.0f}%</span>
            </div>
            <div class="app-verdict" style="color: {tone};">
                <span class="app-verdict-dot" style="background: {dot};"></span>
                {status_text}
            </div>
            <p class="app-note">{html.escape(snapshot.message)}</p>
            <div class="app-meter" role="progressbar" aria-valuemin="0" aria-valuemax="100"
                 aria-valuenow="{width:.0f}">
                <span style="width: {width:.1f}%; background: {bar};"></span>
            </div>
            <div class="app-meter-row">
                <span>spoof {spoof}</span>
                <span>threshold {threshold * 100:.0f}%</span>
            </div>
            <div class="app-stats">
                <div class="app-stat"><span>Confidence</span><strong>{confidence}</strong></div>
                <div class="app-stat"><span>Spoof score</span><strong>{spoof}</strong></div>
                <div class="app-stat"><span>Frame rate</span><strong>{snapshot.fps:.1f}</strong></div>
                <div class="app-stat"><span>Frame</span><strong>{snapshot.frame_index}</strong></div>
            </div>
            <div class="app-panel-foot">
                <span>Age &mdash;</span>
                <span>Emotion &mdash;</span>
                <span>head belum tersedia</span>
            </div>
        </aside>
        """
    )


def render_previews(snapshot: LiveSnapshot | None) -> None:
    if snapshot is None:
        return
    with st.expander("Model input preview (RGB & FFT)", expanded=False):
        col_rgb, col_fft = st.columns(2)
        if snapshot.face_rgb is not None:
            col_rgb.image(snapshot.face_rgb, caption="RGB crop (MobileNetV3 input)")
        else:
            col_rgb.info("No face crop yet.")
        if snapshot.fft_rgb is not None:
            col_fft.image(snapshot.fft_rgb, caption="2D FFT magnitude spectrum")
        else:
            col_fft.info("No FFT spectrum yet.")


def schedule_live_refresh(interval_seconds: float) -> None:
    """Refresh script berkala agar panel status mengikuti worker WebRTC."""
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(interval_seconds * 1000), key="liveness_refresh")
        return
    except ImportError:
        pass
    # Fallback tanpa dependency tambahan: yield ke event loop lalu rerun.
    time.sleep(interval_seconds)
    st.rerun()


def render_model_tab(engine: LivenessEngine | None, engine_error: str | None, model_path: Path) -> None:
    render_html(
        """
        <div class="app-model-head">
            <span class="app-mono">Spesifikasi model</span>
            <h2>MobileNetV3 &middot; spoof classifier</h2>
            <p>Checkpoint dimuat sekali per sesi, lalu dipanggil langsung untuk setiap frame.</p>
        </div>
        """
    )
    if engine is None:
        st.warning(f"Model gagal dimuat: {engine_error or 'unknown error'}")
        st.code(str(model_path), language="text")
        return

    info = engine.describe()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total parameter", f"{info['params']:,}")
    col2.metric("Jumlah input", len(info["inputs"]))
    col3.metric("Jenis stream", info["stream"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Input tensor**")
        st.table(info["inputs"])
    with right:
        st.markdown("**Output tensor**")
        st.table(info["outputs"])
    st.caption(f"Bobot dimuat dari: {info['path']}")


def get_engine(model_path: Path, margin_ratio: float, force_reload: bool) -> tuple[LivenessEngine | None, str | None]:
    """Muat engine sekali; simpan error agar tidak retry setiap rerun."""
    if force_reload:
        load_engine.clear()
        for key in ("engine_error_key", "engine_error_msg", "engine_error_trace"):
            if key in st.session_state:
                del st.session_state[key]

    key = f"{model_path}|{margin_ratio}"
    if not model_path.exists():
        return None, f"Model weights not found: {model_path}"
    if not FACE_MODEL_PATH.exists():
        return None, f"Face detector {FACE_MODEL_PATH.name} not found in the project root."

    size = model_path.stat().st_size
    if size < MIN_VALID_MODEL_BYTES:
        return None, (
            f"File weights {model_path.name} hanya berukuran {size:,} bytes, sehingga tidak berisi "
            "arsitektur maupun bobot (kemungkinan file terpotong). "
            "Export ulang dengan model.save(...); file MobileNetV3 seharusnya berukuran beberapa MB."
        )

    if st.session_state.get("engine_error_key") == key:
        return None, st.session_state.get("engine_error_msg")

    try:
        engine = load_engine(str(model_path), str(FACE_MODEL_PATH), float(margin_ratio))
    except Exception as exc:
        st.session_state.engine_error_key = key
        st.session_state.engine_error_msg = f"{type(exc).__name__}: {exc}"
        st.session_state.engine_error_trace = traceback.format_exc()
        return None, st.session_state.engine_error_msg

    st.session_state.engine_error_key = None
    st.session_state.engine_error_msg = None
    return engine, None


# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="streamlit app · Face Anti-Spoofing",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    render_header()

    config = load_config()
    preprocessing_cfg = config.get("preprocessing", {}) if isinstance(config, dict) else {}
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}

    default_threshold = float(training_cfg.get("threshold", 0.5))
    margin_ratio = float(preprocessing_cfg.get("margin_ratio", 0.15))

    model_files = discover_models()
    preferred_model = resolve_default_model_path(MODELS_DIR)
    default_index = 0
    for index, path in enumerate(model_files):
        if path == preferred_model or path.name == preferred_model.name:
            default_index = index
            break

    with st.sidebar:
        st.header("Kontrol")
        source_options = ["WebRTC (real-time)"]
        if not WEBRTC_AVAILABLE:
            source_options = ["Camera snapshot (OpenCV)"]
        else:
            source_options.append("Camera snapshot (OpenCV)")

        source = st.radio("Sumber video", source_options, index=0)

        if model_files:
            labels = [path.name for path in model_files]
            selected_name = st.selectbox("Bobot model", labels, index=default_index)
            model_path = model_files[labels.index(selected_name)]
        else:
            st.warning(
                "Folder `models/` kosong. Letakkan `model.keras` (disarankan) "
                "atau `model.weights.h5` di sana."
            )
            raw_path = st.text_input("Path bobot", value=str(preferred_model))
            model_path = resolve_path(raw_path)

        st.caption(f"Model: `{model_path}`")

        threshold = st.slider(
            "Ambang spoof",
            min_value=0.05,
            max_value=0.95,
            value=min(max(default_threshold, 0.05), 0.95),
            step=0.05,
            help="Skor di atas ambang dihitung sebagai spoof. Turunkan jika serangan replay lolos.",
        )

        smoothing = st.slider(
            "Smoothing (frame)",
            min_value=1,
            max_value=20,
            value=7,
            help="Moving average dari spoof score agar status tidak berkedip.",
        )

        mirror = st.checkbox("Mirror kamera", value=True)
        show_previews = st.checkbox("Tampilkan crop & FFT", value=False)

        reload_clicked = st.button("Muat ulang model", use_container_width=True)

        if not WEBRTC_AVAILABLE:
            st.error(f"streamlit-webrtc tidak tersedia: {WEBRTC_IMPORT_ERROR}")

    engine_error: str | None = None
    engine: LivenessEngine | None = None

    if reload_clicked:
        load_engine.clear()
        for key in ("engine_error_key", "engine_error_msg", "engine_error_trace"):
            if key in st.session_state:
                del st.session_state[key]

    if model_path.exists():
        with st.spinner("Memuat model anti-spoofing..."):
            engine, engine_error = get_engine(model_path, margin_ratio, reload_clicked)
    else:
        engine_error = f"Model weights not found: {model_path}"

    settings = get_runtime_settings(threshold, smoothing, mirror, show_previews)
    live_state = get_live_state()

    if reload_clicked:
        live_state.reset()
        st.session_state.engine_generation = st.session_state.get("engine_generation", 0) + 1

    engine_generation = st.session_state.get("engine_generation", 0)

    tab_live, tab_model = st.tabs(["Live viewport", "Model"])

    with tab_live:
        left, right = st.columns([3, 2], gap="medium")
        display_snapshot: LiveSnapshot | None = None
        current_snapshot: LiveSnapshot | None = None
        is_live = False

        with left:
            rail_snapshot = live_state.snapshot()

            with st.container(border=False, key="app_viewport"):
                render_viewport_header(rail_snapshot, threshold)
                if source.startswith("WebRTC"):
                    if not WEBRTC_AVAILABLE:
                        st.error("Mode WebRTC tidak tersedia di environment ini.")
                    elif engine is None:
                        st.warning("Model belum siap. Perbaiki error model terlebih dahulu.")
                    else:
                        ctx = webrtc_streamer(
                            key=f"liveness-webrtc-stream-{'mirror' if mirror else 'normal'}",
                            video_processor_factory=lambda: LivenessVideoProcessor(
                                engine, settings, live_state
                            ),
                            media_stream_constraints={
                                "video": {
                                    "width": {"ideal": 1280},
                                    "height": {"ideal": 720},
                                    "frameRate": {"ideal": 24, "max": 30},
                                },
                                "audio": False,
                            },
                            async_processing=True,
                            rtc_configuration={
                                "iceServers": [
                                    {"urls": ["stun:stun.l.google.com:19302"]},
                                    {"urls": ["stun:stun1.l.google.com:19302"]},
                                    {"urls": ["stun:stun2.l.google.com:19302"]},
                                ]
                            },
                        )
                        is_live = bool(getattr(ctx.state, "playing", False))
                        if is_live:
                            display_snapshot = live_state.snapshot()
                else:
                    if engine is None:
                        st.warning("Model belum siap. Perbaiki error model terlebih dahulu.")
                    else:
                        uploaded = st.camera_input(
                            "Ambil satu frame dari kamera (Mode snapshot OpenCV)"
                        )
                        if uploaded is not None:
                            buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                            frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                            if frame_bgr is None:
                                st.error("Frame kamera tidak dapat dibaca.")
                            else:
                                try:
                                    if settings.mirror:
                                        frame_bgr = cv2.flip(frame_bgr, 1)
                                    result = engine.predict(frame_bgr)
                                    display_snapshot = build_snapshot(
                                        result,
                                        result.spoof_score,
                                        settings.threshold,
                                        fps=0.0,
                                        frame_index=1,
                                        window=1,
                                    )
                                    live_state.publish(
                                        display_snapshot,
                                        score=display_snapshot.spoof_score
                                        if display_snapshot.status in ("REAL", "SPOOF")
                                        else None,
                                    )
                                    annotated = draw_overlay(
                                        frame_bgr, display_snapshot, 0.0, settings.threshold
                                    )
                                    st.image(
                                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        caption="Frame + prediction",
                                    )
                                except Exception as exc:
                                    st.error(f"Gagal memproses frame: {exc}")

        with right:
            render_side_panel(
                display_snapshot if display_snapshot is not None else live_state.snapshot(),
                threshold,
                active=display_snapshot is not None or is_live,
            )

            current_snapshot = display_snapshot if display_snapshot is not None else (
                live_state.snapshot() if is_live else None
            )

        if show_previews:
            render_previews(current_snapshot)

        if source.startswith("WebRTC") and WEBRTC_AVAILABLE and engine is not None:
            if is_live:
                schedule_live_refresh(2.0)

    with tab_model:
        render_model_tab(engine, engine_error, model_path)


if __name__ == "__main__":
    main()

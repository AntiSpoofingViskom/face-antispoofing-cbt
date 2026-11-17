"""Role 3 - Web User Interface for Face Anti-Spoofing (Liveness Detection).

Jalankan dari root project::

    streamlit run app/streamlit_app.py

Alur data:
    Webcam (WebRTC thread) -> FacePreprocessor (MediaPipe + 2D FFT)
    -> model MobileNetV3 hasil training Role 2 -> status prediksi real-time.

Catatan desain:
    * Model dimuat sekali lalu di-cache oleh Streamlit (``st.cache_resource``).
    * Inferensi berjalan di thread streamlit-webrtc, bukan di thread UI.
    * Hasil inferensi dibagikan ke UI lewat ``LiveState`` yang thread-safe.
    * Status panel di-refresh otomatis, sedangkan overlay pada video real-time.
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

# ---------------------------------------------------------------------------
# Path & konfigurasi project
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
FACE_MODEL_PATH = ROOT / "blaze_face_short_range.tflite"
DEFAULT_MODEL_PATH = MODELS_DIR / "antispoofing_model.h5"
DEFAULT_INPUT_SIZE = (224, 224)
SUPPORTED_MODEL_EXTENSIONS = ("*.h5", "*.keras", "*.hdf5")

# Streamlit-WebRTC bersifat opsional: jika gagal di-import, UI otomatis
# menawarkan mode snapshot OpenCV agar aplikasi tetap bisa didemokan.
try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - tergantung environment
    av = None  # type: ignore[assignment]
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    class VideoProcessorBase:  # type: ignore[no-redef]
        """Fallback dummy base class ketika streamlit-webrtc tidak tersedia."""

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
    label: str = "Menunggu kamera"
    spoof_score: float = 0.0
    confidence: float = 0.0
    fps: float = 0.0
    frame_index: int = 0
    bbox: tuple[int, int, int, int] | None = None
    message: str = "Tekan START untuk memulai kamera."
    face_rgb: np.ndarray | None = None
    fft_rgb: np.ndarray | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class RuntimeSettings:
    """Pengaturan yang bisa diubah user tanpa me-restart stream WebRTC."""

    threshold: float = 0.5
    smoothing: int = 7
    mirror: bool = True
    predict_interval: float = 0.08
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
            raise FileNotFoundError(f"Bobot model tidak ditemukan: {self.model_path}")
        if not Path(face_model_path).exists():
            raise FileNotFoundError(
                f"Model deteksi wajah tidak ditemukan: {face_model_path}. "
                "Pastikan blaze_face_short_range.tflite ada di root project."
            )

        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        self.input_names = self._resolve_input_names()
        self.num_inputs = len(self.model.inputs)
        self._sizes = [self._input_size(inp) for inp in self.model.inputs]
        self._fft_index = self._resolve_fft_index()

        self.preprocessor = FacePreprocessor(
            model_asset_path=str(face_model_path),
            margin_ratio=margin_ratio,
        )
        self._warmup()

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
        if self.num_inputs >= 2:
            feed: dict[str, np.ndarray] = {}
            for index, name in enumerate(self.input_names):
                image = fft if index == self._fft_index else rgb
                feed[name] = np.expand_dims(image.astype(np.float32), axis=0)
        else:
            feed = np.expand_dims(rgb.astype(np.float32), axis=0)

        output = self.model(feed, training=False)
        score = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(score):
            raise ValueError("Model menghasilkan skor yang tidak valid (NaN/Inf).")
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Jalankan pipeline lengkap untuk satu frame BGR."""
        processed = self.preprocessor.process_frame(frame_bgr, return_bbox=True)
        if processed is None:
            return DetectionResult(ok=False, message="Wajah tidak terdeteksi")

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
            shape = [dim if dim is not None else "None" for dim in tensor.shape]
            inputs.append({"name": name, "shape": shape})
        outputs = []
        for tensor in self.model.outputs:
            shape = [dim if dim is not None else "None" for dim in tensor.shape]
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
STATUS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "REAL": (74, 222, 128),
    "SPOOF": (68, 68, 239),
    "NO_FACE": (11, 158, 245),
    "ERROR": (68, 68, 239),
    "WAITING": (148, 163, 184),
}

STATUS_HEX: dict[str, str] = {
    "REAL": "#22c55e",
    "SPOOF": "#ef4444",
    "NO_FACE": "#f59e0b",
    "ERROR": "#ef4444",
    "WAITING": "#64748b",
}


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
        no_face = "tidak terdeteksi" in result.message.lower()
        return LiveSnapshot(
            status="NO_FACE" if no_face else "ERROR",
            label="Wajah tidak terdeteksi" if no_face else "Terjadi kesalahan",
            fps=round(fps, 1),
            frame_index=frame_index,
            message=result.message,
            face_rgb=result.face_rgb,
            fft_rgb=result.fft_rgb,
        )

    is_spoof = score >= threshold
    confidence = score * 100.0 if is_spoof else (1.0 - score) * 100.0
    label = "PALSU (SPOOF)" if is_spoof else "ASLI (BONAFIDE)"
    window_note = f"rata-rata {max(1, window)} frame" if window > 1 else "prediksi frame ini"
    return LiveSnapshot(
        status="SPOOF" if is_spoof else "REAL",
        label=label,
        spoof_score=float(score),
        confidence=float(confidence),
        fps=round(fps, 1),
        frame_index=frame_index,
        bbox=result.bbox,
        message=f"Skor spoof {score * 100:.1f}% | {window_note} | ambang {threshold * 100:.0f}%",
        face_rgb=result.face_rgb,
        fft_rgb=result.fft_rgb,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    snapshot: LiveSnapshot,
    fps: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Gambar status, confidence, FPS, dan bounding box pada frame."""
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image

    def ascii_only(text: str) -> str:
        # cv2.putText tidak mendukung karakter non-ASCII.
        return text.encode("ascii", "ignore").decode("ascii")

    color = STATUS_COLORS_BGR.get(snapshot.status, STATUS_COLORS_BGR["WAITING"])
    banner_height = max(58, int(height * 0.09))

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    status_text = ascii_only(snapshot.label.upper())
    cv2.putText(
        image,
        status_text,
        (14, int(banner_height * 0.46)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if snapshot.status in ("REAL", "SPOOF"):
        detail = (
            f"spoof {snapshot.spoof_score * 100:.1f}%  |  "
            f"conf {snapshot.confidence:.1f}%  |  ambang {threshold * 100:.0f}%"
        )
    else:
        detail = ascii_only(snapshot.message)
    cv2.putText(
        image,
        detail[:110],
        (14, int(banner_height * 0.86)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if snapshot.bbox is not None:
        x1, y1, x2, y2 = snapshot.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    fps_text = f"{fps:.1f} FPS"
    (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        image,
        fps_text,
        (max(10, width - text_w - 14), height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
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

        now = time.perf_counter()
        if now - self._last_predict_at >= self.settings.predict_interval:
            self._last_predict_at = now
            try:
                result = self.engine.predict(image)
            except Exception as exc:  # jangan matikan stream karena satu frame error
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

        annotated = draw_overlay(
            image,
            self._snapshot,
            self._fps,
            threshold=self.settings.threshold,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    try:
        import yaml

        with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def discover_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    found: set[Path] = set()
    for pattern in SUPPORTED_MODEL_EXTENSIONS:
        try:
            found.update(MODELS_DIR.glob(pattern))
        except OSError:
            continue
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


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


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1250px; }
            .status-card {
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-left: 4px solid #64748b;
                border-radius: 10px;
                padding: 1rem 1.15rem;
                margin-bottom: 1rem;
            }
            .status-eyebrow {
                font-size: 0.72rem; letter-spacing: 0.09em; color: #94a3b8;
                font-weight: 600; text-transform: uppercase;
            }
            .status-row { display: flex; align-items: center; gap: 0.55rem; margin-top: 0.35rem; }
            .status-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
            .status-title { font-size: 1.4rem; font-weight: 600; }
            .status-sub { color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title("Face Anti-Spoofing")
    st.caption(
        "Deteksi wajah asli (bonafide) atau palsu (spoof) secara real-time melalui webcam."
    )


def render_status_card(snapshot: LiveSnapshot | None, threshold: float, active: bool) -> None:
    if snapshot is None or not active:
        snapshot = LiveSnapshot(
            status="WAITING",
            label="Kamera belum aktif",
            message="Tekan START pada panel kamera untuk memulai deteksi.",
        )

    color = STATUS_HEX.get(snapshot.status, STATUS_HEX["WAITING"])

    st.markdown(
        f"""
        <div class="status-card" style="border-left-color: {color};">
            <div class="status-eyebrow">Status</div>
            <div class="status-row">
                <span class="status-dot" style="background: {color};"></span>
                <span class="status-title">{html.escape(snapshot.label)}</span>
            </div>
            <div class="status-sub">{html.escape(snapshot.message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Skor spoof", f"{snapshot.spoof_score * 100:.1f}%")
    col2.metric("Keyakinan", f"{snapshot.confidence:.1f}%")
    col3.metric("FPS", f"{snapshot.fps:.1f}")

    st.progress(
        min(max(float(snapshot.spoof_score), 0.0), 1.0),
        text=f"Ambang batas spoof: {threshold * 100:.0f}%",
    )
    if snapshot.status in ("REAL", "SPOOF"):
        st.caption(
            f"Frame {snapshot.frame_index} | skor di atas ambang dianggap PALSU, "
            "skor di bawah ambang dianggap ASLI."
        )


def render_history(live_state: LiveState, threshold: float) -> None:
    history = live_state.history()
    if len(history) < 2:
        return
    import pandas as pd

    frame = pd.DataFrame(history, columns=["Waktu", "Skor spoof"])
    frame["Waktu"] = pd.to_datetime(frame["Waktu"], unit="s")
    frame = frame.set_index("Waktu")
    st.markdown("##### Riwayat skor spoof")
    st.line_chart(frame, height=180)
    st.caption(
        f"Ambang spoof saat ini {threshold * 100:.0f}%. "
        "Skor di atas ambang diklasifikasikan PALSU."
    )


def render_previews(snapshot: LiveSnapshot | None) -> None:
    if snapshot is None:
        return
    with st.expander("Pratinjau input model (RGB & FFT)", expanded=False):
        col_rgb, col_fft = st.columns(2)
        if snapshot.face_rgb is not None:
            col_rgb.image(snapshot.face_rgb, caption="RGB crop (untuk MobileNetV3)")
        else:
            col_rgb.info("Belum ada crop wajah.")
        if snapshot.fft_rgb is not None:
            col_fft.image(snapshot.fft_rgb, caption="2D FFT magnitude spectrum")
        else:
            col_fft.info("Belum ada spektrum FFT.")


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
    st.subheader("Model & bobot Role 2")
    if engine is None:
        st.warning(f"Model belum berhasil dimuat: {engine_error or 'tidak diketahui'}")
        st.code(str(model_path), language="text")
        return

    info = engine.describe()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total parameter", f"{info['params']:,}")
    col2.metric("Jumlah input", len(info["inputs"]))
    col3.metric("Jenis stream", info["stream"])

    st.markdown("##### Input tensor")
    st.table(info["inputs"])
    st.markdown("##### Output tensor")
    st.table(info["outputs"])
    st.caption(f"Bobot dimuat dari: {info['path']}")
    st.info(
        "Model dimuat dengan `compile=False` lalu dipanggil langsung "
        "(`model(inputs, training=False)`) agar kompatibel dengan checkpoint `.h5`/`.keras`."
    )


def get_engine(model_path: Path, margin_ratio: float, force_reload: bool) -> tuple[LivenessEngine | None, str | None]:
    """Muat engine sekali; simpan error agar tidak retry setiap rerun."""
    if force_reload:
        load_engine.clear()
        for key in ("engine_error_key", "engine_error_msg", "engine_error_trace"):
            if key in st.session_state:
                del st.session_state[key]

    key = f"{model_path}|{margin_ratio}"
    if not model_path.exists():
        return None, f"Bobot model tidak ditemukan di {model_path}"
    if not FACE_MODEL_PATH.exists():
        return None, f"File wajah {FACE_MODEL_PATH.name} tidak ditemukan di root project."

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
        page_title="Face Anti-Spoofing",
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
    default_index = 0
    for index, path in enumerate(model_files):
        if path.name == DEFAULT_MODEL_PATH.name:
            default_index = index
            break

    with st.sidebar:
        st.header("Kontrol")
        source_options = ["WebRTC (real-time)"]
        if not WEBRTC_AVAILABLE:
            source_options = ["Snapshot kamera (OpenCV)"]
        else:
            source_options.append("Snapshot kamera (OpenCV)")
        source = st.radio("Sumber video", source_options, index=0)

        if model_files:
            labels = [path.name for path in model_files]
            selected_name = st.selectbox("Bobot model (Role 2)", labels, index=default_index)
            model_path = model_files[labels.index(selected_name)]
        else:
            st.warning("Folder `models/` belum berisi bobot. Letakkan `antispoofing_model.h5` di sana.")
            raw_path = st.text_input("Path bobot model", value=str(DEFAULT_MODEL_PATH))
            model_path = resolve_path(raw_path)

        st.caption(f"Model: `{model_path}`")

        threshold = st.slider(
            "Ambang spoof",
            min_value=0.05,
            max_value=0.95,
            value=min(max(default_threshold, 0.05), 0.95),
            step=0.05,
            help="Skor spoof di atas ambang dianggap PALSU. Turunkan bila replay attack lolos.",
        )
        smoothing = st.slider(
            "Smoothing (jumlah frame)",
            min_value=1,
            max_value=20,
            value=7,
            help="Moving average skor spoof agar status tidak berkedip.",
        )
        mirror = st.checkbox("Mirror kamera", value=True)
        show_previews = st.checkbox("Tampilkan pratinjau crop & FFT", value=False)
        reload_clicked = st.button("Muat ulang model", use_container_width=True)

        if not WEBRTC_AVAILABLE:
            st.error(f"streamlit-webrtc tidak tersedia: {WEBRTC_IMPORT_ERROR}")
        st.divider()
        st.caption("Role 3 - UI / Webcam / Integrasi pipeline")

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
        engine_error = f"Bobot model tidak ditemukan di {model_path}"

    settings = get_runtime_settings(threshold, smoothing, mirror, show_previews)
    live_state = get_live_state()

    if reload_clicked:
        live_state.reset()
        st.session_state.engine_generation = st.session_state.get("engine_generation", 0) + 1
    engine_generation = st.session_state.get("engine_generation", 0)

    if engine_error is not None:
        st.error(f"Model gagal dimuat: {engine_error}")
        st.info(
            "Pastikan Role 2 sudah menyimpan bobot terbaik ke `models/` "
            "(mis. `models/antispoofing_model.h5`) dan file "
            "`blaze_face_short_range.tflite` ada di root project."
        )
        if st.session_state.get("engine_error_trace"):
            with st.expander("Detail error"):
                st.code(st.session_state.engine_error_trace, language="text")

    tab_live, tab_model = st.tabs(["Live Detection", "Model"])

    with tab_live:
        left, right = st.columns([3, 2], gap="large")
        display_snapshot: LiveSnapshot | None = None
        is_live = False

        with left:
            if source.startswith("WebRTC"):
                if not WEBRTC_AVAILABLE:
                    st.error("Mode WebRTC tidak tersedia di environment ini.")
                elif engine is None:
                    st.warning("Model belum siap, perbaiki error model terlebih dahulu.")
                else:
                    ctx = webrtc_streamer(
                        key=f"liveness-webrtc-{model_path.name}-{int(margin_ratio * 100)}-{engine_generation}",
                        video_processor_factory=lambda: LivenessVideoProcessor(
                            engine, settings, live_state
                        ),
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 1280},
                                "height": {"ideal": 720},
                                "frameRate": {"ideal": 24},
                            },
                            "audio": False,
                        },
                        async_processing=True,
                        rtc_configuration={
                            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                        },
                    )
                    is_live = bool(getattr(ctx.state, "playing", False))
                    if is_live:
                        display_snapshot = live_state.snapshot()
                        st.caption(
                            "Kamera aktif. Overlay dan panel status diperbarui otomatis. "
                            "Tekan STOP untuk menghentikan."
                        )
                    else:
                        st.info(
                            "Tekan **START** di atas dan izinkan akses kamera. "
                            "Video dan prediksi akan muncul dalam beberapa detik."
                        )
            else:
                if engine is None:
                    st.warning("Model belum siap, perbaiki error model terlebih dahulu.")
                else:
                    uploaded = st.camera_input(
                        "Ambil satu frame dari kamera (mode OpenCV snapshot)"
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
                                    caption="Frame + hasil prediksi",
                                )
                            except Exception as exc:
                                st.error(f"Gagal memproses frame: {exc}")

        with right:
            render_status_card(
                display_snapshot if display_snapshot is not None else live_state.snapshot(),
                threshold,
                active=display_snapshot is not None or is_live,
            )
            if engine is not None:
                engine_info = engine.describe()
                st.caption(
                    f"Model aktif: **{Path(engine_info['path']).name}** | "
                    f"{engine_info['stream']} | ambang {threshold * 100:.0f}%"
                )

        current_snapshot = display_snapshot if display_snapshot is not None else (
            live_state.snapshot() if is_live else None
        )
        if current_snapshot is not None and current_snapshot.status in ("REAL", "SPOOF"):
            render_history(live_state, threshold)
        if show_previews:
            render_previews(current_snapshot)

        if source.startswith("WebRTC") and WEBRTC_AVAILABLE and engine is not None:
            schedule_live_refresh(0.4 if is_live else 1.0)

    with tab_model:
        render_model_tab(engine, engine_error, model_path)

    st.divider()
    st.caption(
        "Face Anti-Spoofing | Streamlit UI, WebRTC capture, integrasi FacePreprocessor + MobileNetV3."
    )


if __name__ == "__main__":
    main()

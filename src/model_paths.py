"""Penentuan file model anti-spoofing dari folder ``models/``.

Cukup letakkan file model di ``models/`` lalu aplikasi akan menemukannya
otomatis. Urutan prioritas pemilihan model default:

1. Path dari environment variable ``FAS_MODEL_PATH`` (kalau diisi).
2. ``model.keras`` / ``antispoofing_model.keras`` (full model, disarankan).
3. File ``*.keras`` lain, yang paling baru diubah.
4. ``model.weights.h5`` (bobot saja, arsitektur dari ``config.json``).
5. ``*.h5`` / ``*.hdf5`` lain, yang paling baru diubah.
6. ``models/model.weights.h5`` sebagai fallback (walau belum ada filenya).

File yang ukurannya di bawah ``MIN_VALID_BYTES`` dilewati saat memilih model
default karena biasanya hanya header/unduhan yang gagal.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pola file yang dikenali (urutan tidak menentukan prioritas).
MODEL_EXTENSIONS = ("*.keras", "*.h5", "*.hdf5")

# Nama yang diprioritaskan kalau ada lebih dari satu file .keras.
PREFERRED_KERAS_NAMES = ("model.keras", "antispoofing_model.keras")
PREFERRED_WEIGHTS_NAME = "model.weights.h5"
FALLBACK_NAME = "model.weights.h5"

# File model valid minimal berukuran ~1 KB (header HDF5/zip kosong lebih kecil).
MIN_VALID_BYTES = 1024


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def discover_model_files(models_dir: str | Path) -> list[Path]:
    """Kembalikan semua file model yang didukung, paling baru lebih dulu."""
    directory = Path(models_dir)
    if not directory.is_dir():
        return []

    found: dict[Path, float] = {}
    for pattern in MODEL_EXTENSIONS:
        try:
            matches = directory.glob(pattern)
        except OSError:
            continue
        for path in matches:
            if path.is_file():
                found[path.resolve()] = _mtime(path)

    return sorted(found, key=found.get, reverse=True)


def _pick(files: list[Path], names: tuple[str, ...]) -> Path | None:
    for name in names:
        for path in files:
            if path.name.lower() == name:
                return path
    return None


def resolve_default_model_path(
    models_dir: str | Path,
    fallback_name: str = FALLBACK_NAME,
) -> Path:
    """Tentukan file model yang dipakai otomatis oleh app/inference."""
    directory = Path(models_dir)

    override = os.environ.get("FAS_MODEL_PATH", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = directory.parent / override_path
        if override_path.exists():
            return override_path

    files = [path for path in discover_model_files(directory) if _size(path) >= MIN_VALID_BYTES]

    keras_files = [path for path in files if path.suffix.lower() == ".keras"]
    preferred_keras = _pick(keras_files, PREFERRED_KERAS_NAMES)
    if preferred_keras is not None:
        return preferred_keras
    if keras_files:
        return keras_files[0]

    weights = _pick(files, (PREFERRED_WEIGHTS_NAME,))
    if weights is not None:
        return weights

    if files:
        return files[0]

    return directory / fallback_name

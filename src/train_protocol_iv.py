"""Train dual-stream model with OULU-NPU Protocol IV splits + spoof replay."""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_model

BASE_DIR = Path(__file__).resolve().parents[1]
SEED = 123
IMG_SIZE = (224, 224)


def read_split(path):
    entries = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                raw_path, label = line.split("\t")
                entries.append((Path(raw_path).name, 0 if label == "bonafide" else 1))
    return entries


def processed_index(root, label, split="train"):
    directory = root / "data" / "processed" / "rgb" / split / label
    index = {}
    for path in directory.glob("*.jpg"):
        # Remove prefix untuk matching
        stem = path.stem.removeprefix(f"{label}_img_")
        index[stem] = path
    return index


def load_entries(entries, rgb_index, fft_root, label_name, split="train"):
    rgb_images, fft_images, labels = [], [], []
    fft_dir = fft_root / "data" / "processed" / "fft" / split / label_name
    fft_index = {path.name: path for path in fft_dir.glob("*.jpg")}

    for name, label in entries:
        stem = Path(name).stem
        rgb_path = rgb_index.get(stem)
        if rgb_path is None:
            continue
        fft_path = fft_index.get(rgb_path.name)
        if fft_path is None:
            continue
        rgb = cv2.imread(str(rgb_path))
        fft = cv2.imread(str(fft_path))
        if rgb is None or fft is None:
            continue
        rgb = cv2.resize(rgb, IMG_SIZE)
        fft = cv2.resize(fft, IMG_SIZE)
        rgb_images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        fft_images.append(cv2.cvtColor(fft, cv2.COLOR_BGR2RGB))
        labels.append(label)
    if not labels:
        raise RuntimeError(f"Tidak ada pasangan RGB/FFT ditemukan untuk label {label_name}.")
    return rgb_images, fft_images, labels


def load_fold(fold, root):
    split_root = root / "data" / "splits"
    train_entries = read_split(split_root / f"fold_{fold}_train.txt")
    dev_entries = read_split(split_root / f"fold_{fold}_dev.txt")

    # Custom bonafide tidak punya kamera OULU; masukkan hanya ke train.
    custom_dir = root / "data" / "processed" / "rgb" / "train" / "bonafide"
    custom_files = [p.name for p in custom_dir.glob("bonafide_img_*")]
    train_entries.extend((name.removeprefix("bonafide_img_"), 0) for name in custom_files)

    # Tambahkan spoof replay TRAIN ke training (file dengan _replay_train_)
    replay_dir = root / "data" / "processed" / "rgb" / "train" / "spoof"
    if replay_dir.exists():
        replay_files = [p.name for p in replay_dir.glob("*_replay_train_*")]
        train_entries.extend((name, 1) for name in replay_files)  # label 1 = spoof

    rgb_index = {}
    for label in ("bonafide", "spoof"):
        rgb_index.update(processed_index(root, label, split="train"))

    def load(entries):
        rgb, fft, labels = [], [], []
        for label_name, label_value in (("bonafide", 0), ("spoof", 1)):
            selected = [(name, label) for name, label in entries if label == label_value]
            r, f, y = load_entries(selected, rgb_index, root, label_name, split="train")
            rgb.extend(r); fft.extend(f); labels.extend(y)
        return np.asarray(rgb, dtype=np.float32), np.asarray(fft, dtype=np.float32), np.asarray(labels, dtype=np.float32)

    return load(train_entries), load(dev_entries)


def make_dataset(data, batch_size, shuffle):
    rgb, fft, labels = data
    ds = tf.data.Dataset.from_tensor_slices(({"rgb_input": rgb, "fft_input": fft}, labels))
    return ds.shuffle(len(labels), seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE) if shuffle else ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_fold(fold, epochs, batch_size, root, output_dir):
    train_data, dev_data = load_fold(fold, root)
    train_ds = make_dataset(train_data, batch_size, True)
    dev_ds = make_dataset(dev_data, batch_size, False)
    labels = train_data[2].astype(int)
    counts = np.bincount(labels, minlength=2)
    class_weight = {i: len(labels) / (2 * count) for i, count in enumerate(counts) if count}

    model = build_model()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(output_dir / f"dual_fold_{fold}.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    print(f"Fold {fold}: train={len(labels)}, dev={len(dev_data[2])}, class_weight={class_weight}")
    model.fit(train_ds, validation_data=dev_ds, epochs=epochs, callbacks=[checkpoint], class_weight=class_weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="0 = all folds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    folds = range(1, 7) if args.fold == 0 else [args.fold]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "models"
    for fold in folds:
        train_fold(fold, epochs, batch_size, BASE_DIR, output_dir)


if __name__ == "__main__":
    main()

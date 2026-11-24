"""Train FFT-only model with OULU-NPU Protocol IV splits + spoof replay."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint

from model_fft_only import build_model_fft

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


def get_processed_index(root, label_name, split="train"):
    fft_dir = root / "data" / "processed" / "fft" / split / label_name
    index = {}
    if fft_dir.exists():
        for path in fft_dir.glob("*.jpg"):
            index[path.stem] = path
            index[path.stem.removeprefix(f"{label_name}_img_")] = path
    return index


def load_fold_data(fold, root):
    split_dir = root / "data" / "splits"
    train_entries = read_split(split_dir / f"fold_{fold}_train.txt")
    dev_entries = read_split(split_dir / f"fold_{fold}_dev.txt")

    custom_dir = root / "data" / "processed" / "fft" / "train" / "bonafide"
    if custom_dir.exists():
        train_entries.extend((p.name, 0) for p in custom_dir.glob("bonafide_img_*"))

    replay_dir = root / "data" / "processed" / "fft" / "train" / "spoof"
    if replay_dir.exists():
        train_entries.extend((p.name, 1) for p in replay_dir.glob("*_replay_train_*"))

    fft_index = {}
    for label_name in ("bonafide", "spoof"):
        fft_index.update(get_processed_index(root, label_name, split="train"))

    def load_images(entries):
        images, labels = [], []
        for filename, label in entries:
            stem = Path(filename).stem.removeprefix("bonafide_img_").removeprefix("spoof_img_")
            found_path = (
                fft_index.get(stem)
                or fft_index.get(f"bonafide_img_{stem}")
                or fft_index.get(f"spoof_img_{stem}")
            )
            if found_path is None:
                continue
            img = cv2.imread(str(found_path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
            labels.append(label)
        return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.float32)

    train_data = load_images(train_entries)
    dev_data = load_images(dev_entries)
    if len(train_data[1]) == 0 or len(dev_data[1]) == 0:
        raise ValueError(f"Fold {fold}: tidak ada gambar FFT ditemukan!")
    return train_data, dev_data


def make_dataset(images, labels, batch_size, shuffle_data=True):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle_data:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_fold(fold, root, epochs, batch_size, output_dir):
    print(f"\n=== FFT-Only Fold {fold} ===")
    (train_images, train_labels), (dev_images, dev_labels) = load_fold_data(fold, root)
    labels_int = train_labels.astype(int)
    counts = np.bincount(labels_int, minlength=2)
    class_weight = {i: len(labels_int) / (2 * c) for i, c in enumerate(counts) if c}
    print(f"Train: {len(labels_int)} | Dev: {len(dev_labels)} | Class weight: {class_weight}")

    train_ds = make_dataset(train_images, train_labels, batch_size, shuffle_data=True)
    dev_ds = make_dataset(dev_images, dev_labels, batch_size, shuffle_data=False)

    model = build_model_fft()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"fft_fold_{fold}.keras"
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=epochs,
        callbacks=[checkpoint],
        class_weight=class_weight,
    )
    print(f"Model disimpan ke: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=0, help="0 = semua fold (1-6)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs or config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    output_dir = Path(args.output_dir) if args.output_dir else (BASE_DIR / "models")
    folds = range(1, 7) if args.fold == 0 else [args.fold]

    for fold in folds:
        train_fold(fold, BASE_DIR, epochs, batch_size, output_dir)
    print("\n✅ Training FFT-only selesai!")


if __name__ == "__main__":
    main()

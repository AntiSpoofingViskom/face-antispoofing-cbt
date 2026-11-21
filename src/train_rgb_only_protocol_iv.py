"""Train RGB-only baseline model with OULU-NPU Protocol IV splits."""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint

from model_rgb_only import build_model_rgb

BASE_DIR = Path(__file__).resolve().parents[1]
SEED = 123


def read_split(path):
    """Baca file split, return list (filename, label_binary)."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            raw_path, label = parts
            filename = Path(raw_path).name
            label_binary = 0 if label == 'bonafide' else 1
            entries.append((filename, label_binary))
    return entries


def get_processed_index(root, label_name):
    """Buat index file RGB yang ada di processed folder."""
    rgb_dir = root / 'data' / 'processed' / 'rgb' / 'train' / label_name
    index = {}
    if rgb_dir.exists():
        for path in rgb_dir.glob('*.jpg'):
            stem = path.stem
            index[stem] = path
    return index


def load_fold_data(fold, root):
    """Load train dan dev set untuk satu fold."""
    split_dir = root / 'data' / 'splits'
    train_file = split_dir / f'fold_{fold}_train.txt'
    dev_file = split_dir / f'fold_{fold}_dev.txt'

    train_entries = read_split(train_file)
    dev_entries = read_split(dev_file)

    # Buat index semua file RGB yang sudah dipreprocess
    rgb_index = {}
    for label_name in ['bonafide', 'spoof']:
        rgb_index.update(get_processed_index(root, label_name))

    def load_images(entries):
        """Load gambar dari entries."""
        images, labels = [], []
        for filename, label in entries:
            # Hapus prefix jika ada
            stem = filename.replace('bonafide_img_', '').replace('spoof_img_', '')
            stem = Path(stem).stem

            # Cari di index
            found_path = None
            for key, path in rgb_index.items():
                if stem in key or key in stem:
                    found_path = path
                    break

            if found_path is None:
                continue

            img = cv2.imread(str(found_path))
            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            images.append(img_rgb)
            labels.append(label)

        return np.array(images), np.array(labels)

    train_images, train_labels = load_images(train_entries)
    dev_images, dev_labels = load_images(dev_entries)

    if len(train_labels) == 0 or len(dev_labels) == 0:
        raise ValueError(f"Fold {fold}: tidak ada gambar ditemukan!")

    return (train_images, train_labels), (dev_images, dev_labels)


def make_dataset(images, labels, batch_size, shuffle_data=True):
    """Buat tf.data.Dataset dari images dan labels."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle_data:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_fold(fold, root, config, output_dir):
    """Training RGB-only model untuk satu fold."""
    print(f"\n=== Fold {fold} ===")

    # Load data
    (train_images, train_labels), (dev_images, dev_labels) = load_fold_data(fold, root)
    print(f"Train: {len(train_labels)} | Dev: {len(dev_labels)}")

    # Class weight
    unique, counts = np.unique(train_labels, return_counts=True)
    class_weight = {int(u): len(train_labels) / (len(unique) * c) for u, c in zip(unique, counts)}
    print(f"Class weight: {class_weight}")

    # Dataset
    batch_size = config['training']['batch_size']
    train_ds = make_dataset(train_images, train_labels, batch_size, shuffle_data=True)
    dev_ds = make_dataset(dev_images, dev_labels, batch_size, shuffle_data=False)

    # Model
    model = build_model_rgb()

    # Checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f'rgb_fold_{fold}.keras'
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Training
    epochs = config['training']['epochs']
    print(f"Training RGB-only fold {fold} ({epochs} epochs)...")
    model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=epochs,
        callbacks=[checkpoint],
        class_weight=class_weight
    )

    print(f"Model disimpan ke: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=0, help='0 = semua fold, atau fold spesifik (1-6)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) if args.output_dir else (BASE_DIR / 'models')

    folds = range(1, 7) if args.fold == 0 else [args.fold]

    for fold in folds:
        train_fold(fold, BASE_DIR, config, output_dir)

    print("\n✅ Training selesai!")


if __name__ == '__main__':
    main()

"""
Evaluasi model Anti-Spoofing dengan Protocol IV OULU-NPU.
- Cari threshold optimal (EER) dari dev set
- Hitung APCER, BPCER, ACER pada test set
- Bandingkan model Dual-Stream vs RGB Only
"""
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tabulate import tabulate

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from preprocessing import FacePreprocessor

SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
IMG_SIZE = (224, 224)


def load_split(split_file):
    """Baca file split, return list (path, label)."""
    entries = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.split('\t')
            entries.append((path, 0 if label == 'bonafide' else 1))
    return entries


def preprocess_image(img_path, preprocessor):
    """Load gambar, crop wajah, return (rgb_224, fft_224) atau None."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    result = preprocessor.process_frame(img)
    if result is None:
        return None
    rgb_face, fft_spectrum = result
    rgb_resized = cv2.resize(rgb_face, IMG_SIZE).astype(np.float32)
    fft_resized = cv2.resize(fft_spectrum, IMG_SIZE).astype(np.float32)
    return rgb_resized, fft_resized


def predict_dual(model, rgb_images, fft_images):
    """Prediksi dengan model dual-stream."""
    rgb_batch = np.array(rgb_images)
    fft_batch = np.array(fft_images)
    preds = model.predict({'rgb_input': rgb_batch, 'fft_input': fft_batch}, verbose=0)
    return preds.flatten()


def predict_rgb_only(model, rgb_images):
    """Prediksi dengan model RGB only."""
    rgb_batch = np.array(rgb_images)
    preds = model.predict(rgb_batch, verbose=0)
    return preds.flatten()


def get_scores(model, entries, preprocessor, model_type='dual'):
    """Dapatkan skor prediksi dan label untuk semua entry."""
    scores = []
    labels = []
    rgb_batch = []
    fft_batch = []
    label_batch = []

    for path, label in entries:
        result = preprocess_image(path, preprocessor)
        if result is None:
            continue
        rgb, fft = result
        rgb_batch.append(rgb)
        fft_batch.append(fft)
        label_batch.append(label)

        # Prediksi per batch 32
        if len(rgb_batch) == 32:
            if model_type == 'dual':
                preds = predict_dual(model, rgb_batch, fft_batch)
            else:
                preds = predict_rgb_only(model, rgb_batch)
            scores.extend(preds)
            labels.extend(label_batch)
            rgb_batch, fft_batch, label_batch = [], [], []

    # Sisa batch
    if rgb_batch:
        if model_type == 'dual':
            preds = predict_dual(model, rgb_batch, fft_batch)
        else:
            preds = predict_rgb_only(model, rgb_batch)
        scores.extend(preds)
        labels.extend(label_batch)

    return np.array(scores), np.array(labels)


def find_eer_threshold(scores, labels):
    """Cari threshold pada Equal Error Rate (EER) dari dev set."""
    thresholds = np.arange(0.0, 1.01, 0.001)
    min_diff = float('inf')
    best_thresh = 0.5

    for t in thresholds:
        # APCER: attack samples classified as bonafide
        attack_mask = labels == 1
        bonafide_mask = labels == 0

        if attack_mask.sum() == 0 or bonafide_mask.sum() == 0:
            continue

        apcer = (scores[attack_mask] < t).sum() / attack_mask.sum()
        bpcer = (scores[bonafide_mask] >= t).sum() / bonafide_mask.sum()

        diff = abs(apcer - bpcer)
        if diff < min_diff:
            min_diff = diff
            best_thresh = t

    return best_thresh


def compute_metrics(scores, labels, threshold):
    """Hitung APCER, BPCER, ACER dengan threshold tertentu."""
    attack_mask = labels == 1
    bonafide_mask = labels == 0

    if attack_mask.sum() == 0 or bonafide_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    # APCER: proporsi attack yang dikira bonafide (skor < threshold)
    apcer = (scores[attack_mask] < threshold).sum() / attack_mask.sum()
    # BPCER: proporsi bonafide yang dikira attack (skor >= threshold)
    bpcer = (scores[bonafide_mask] >= threshold).sum() / bonafide_mask.sum()
    # ACER: rata-rata
    acer = (apcer + bpcer) / 2.0

    return apcer, bpcer, acer


def evaluate_model(model, model_type, preprocessor, num_folds=6):
    """Evaluasi model pada semua fold Protocol IV."""
    results = []

    for fold in range(1, num_folds + 1):
        dev_file = os.path.join(SPLITS_DIR, f'fold_{fold}_dev.txt')
        test_file = os.path.join(SPLITS_DIR, f'fold_{fold}_test.txt')

        if not os.path.exists(dev_file) or not os.path.exists(test_file):
            print(f"  Fold {fold}: file split tidak ditemukan, skip.")
            continue

        # Prediksi dev set → cari threshold EER
        dev_entries = load_split(dev_file)
        dev_scores, dev_labels = get_scores(model, dev_entries, preprocessor, model_type)
        threshold = find_eer_threshold(dev_scores, dev_labels)

        # Prediksi test set → hitung metrik
        test_entries = load_split(test_file)
        test_scores, test_labels = get_scores(model, test_entries, preprocessor, model_type)
        apcer, bpcer, acer = compute_metrics(test_scores, test_labels, threshold)

        results.append({
            'fold': fold,
            'threshold': threshold,
            'apcer': apcer,
            'bpcer': bpcer,
            'acer': acer,
        })
        print(f"  Fold {fold}: threshold={threshold:.3f} | APCER={apcer:.4f} | BPCER={bpcer:.4f} | ACER={acer:.4f}")

    return results


def main():
    preprocessor = FacePreprocessor(
        model_asset_path=os.path.join(BASE_DIR, 'blaze_face_short_range.tflite'),
        margin_ratio=0.15
    )

    dual_model_path = os.path.join(BASE_DIR, 'models', 'antispoofing_model.keras')
    rgb_model_path = os.path.join(BASE_DIR, 'models', 'antispoofing_rgb_only.keras')

    all_results = {}

    # --- Evaluasi Dual-Stream ---
    if os.path.exists(dual_model_path):
        print("\n=== Evaluasi Model Dual-Stream (RGB + FFT) ===")
        dual_model = tf.keras.models.load_model(dual_model_path)
        all_results['dual'] = evaluate_model(dual_model, 'dual', preprocessor)
    else:
        print(f"Model dual-stream tidak ditemukan: {dual_model_path}")

    # --- Evaluasi RGB Only ---
    if os.path.exists(rgb_model_path):
        print("\n=== Evaluasi Model RGB Only (Baseline) ===")
        rgb_model = tf.keras.models.load_model(rgb_model_path)
        all_results['rgb'] = evaluate_model(rgb_model, 'rgb', preprocessor)
    else:
        print(f"Model RGB only tidak ditemukan: {rgb_model_path}")

    # --- Tabel Perbandingan ---
    if all_results:
        print("\n=== PERBANDINGAN HASIL ===")
        table = []
        for name, results in all_results.items():
            if results:
                avg_apcer = np.mean([r['apcer'] for r in results])
                avg_bpcer = np.mean([r['bpcer'] for r in results])
                avg_acer = np.mean([r['acer'] for r in results])
                table.append([name.upper(), f"{avg_apcer:.4f}", f"{avg_bpcer:.4f}", f"{avg_acer:.4f}"])

        print(tabulate(table, headers=['Model', 'Avg APCER', 'Avg BPCER', 'Avg ACER'], tablefmt='grid'))


if __name__ == '__main__':
    main()

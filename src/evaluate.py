"""
Evaluasi model Anti-Spoofing dengan Protocol IV OULU-NPU (6 fold):
- Mendukung 3 tipe model per-fold: RGB-Only, FFT-Only, Dual-Stream (RGB + FFT)
- Mencari threshold optimal (EER) pada dev set tiap fold
- Menghitung APCER, BPCER, ACER, Accuracy, Precision, Recall, F1, dan Confusion Matrix pada test set
- Menyimpan tabel CSV dan gambar Confusion Matrix ke folder reports/
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parents[1]
IMG_SIZE = (224, 224)


def read_split(split_file):
    entries = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_path, label = line.split("\t")
                entries.append((Path(raw_path).name, 0 if label == "bonafide" else 1))
    return entries


def build_processed_indices(data_root, split="train"):
    rgb_index, fft_index = {}, {}
    for label in ("bonafide", "spoof"):
        rgb_dir = data_root / "processed" / "rgb" / split / label
        fft_dir = data_root / "processed" / "fft" / split / label
        if rgb_dir.exists():
            for p in rgb_dir.glob("*.jpg"):
                rgb_index[p.stem] = p
                rgb_index[p.stem.removeprefix(f"{label}_img_")] = p
        if fft_dir.exists():
            for p in fft_dir.glob("*.jpg"):
                fft_index[p.stem] = p
                fft_index[p.stem.removeprefix(f"{label}_img_")] = p
    return rgb_index, fft_index


def load_split_arrays(entries, rgb_index, fft_index):
    rgb_list, fft_list, labels = [], [], []
    for name, label in entries:
        stem = Path(name).stem.removeprefix("bonafide_img_").removeprefix("spoof_img_")
        prefix = "bonafide_img_" if label == 0 else "spoof_img_"
        rgb_path = rgb_index.get(stem) or rgb_index.get(f"{prefix}{stem}")
        fft_path = fft_index.get(stem) or fft_index.get(f"{prefix}{stem}")
        if rgb_path is None or fft_path is None:
            continue
        rgb = cv2.imread(str(rgb_path))
        fft = cv2.imread(str(fft_path))
        if rgb is None or fft is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(rgb, IMG_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32)
        fft = cv2.cvtColor(cv2.resize(fft, IMG_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_list.append(rgb)
        fft_list.append(fft)
        labels.append(label)
    return np.asarray(rgb_list), np.asarray(fft_list), np.asarray(labels, dtype=np.int32)


def predict_scores(model, model_type, rgb_arr, fft_arr):
    if model_type == "dual":
        preds = model.predict({"rgb_input": rgb_arr, "fft_input": fft_arr}, batch_size=32, verbose=0)
    elif model_type == "rgb":
        preds = model.predict(rgb_arr, batch_size=32, verbose=0)
    elif model_type == "fft":
        preds = model.predict(fft_arr, batch_size=32, verbose=0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return preds.flatten()


def find_eer_threshold(scores, labels):
    thresholds = np.linspace(0.0, 1.0, 1001)
    attack_mask = labels == 1
    bonafide_mask = labels == 0
    if attack_mask.sum() == 0 or bonafide_mask.sum() == 0:
        return 0.5
    best_thresh, min_diff = 0.5, float("inf")
    for t in thresholds:
        apcer = (scores[attack_mask] < t).mean()
        bpcer = (scores[bonafide_mask] >= t).mean()
        diff = abs(apcer - bpcer)
        if diff < min_diff:
            min_diff = diff
            best_thresh = float(t)
    return best_thresh


def compute_all_metrics(scores, labels, threshold):
    # Label convention: 0 = Bonafide (Real), 1 = Spoof (Attack)
    preds = (scores >= threshold).astype(np.int32)
    bonafide_mask = labels == 0
    attack_mask = labels == 1

    tn = int(((preds == 0) & bonafide_mask).sum())  # Bonafide predicted Bonafide
    fp = int(((preds == 1) & bonafide_mask).sum())  # Bonafide predicted Spoof
    fn = int(((preds == 0) & attack_mask).sum())    # Spoof predicted Bonafide
    tp = int(((preds == 1) & attack_mask).sum())    # Spoof predicted Spoof

    apcer = fn / max(1, (fn + tp))
    bpcer = fp / max(1, (fp + tn))
    acer = (apcer + bpcer) / 2.0
    acc = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))

    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["Bonafide (0)", "Spoof (1)"]
    ax.set_xticks([0, 1], labels=classes)
    ax.set_yticks([0, 1], labels=classes)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_model_path(model_type, fold, models_dir, specific_dir):
    search_dirs = [Path(specific_dir)] if specific_dir else []
    search_dirs.extend([Path(models_dir), Path(models_dir) / model_type])
    for d in search_dirs:
        candidate = d / f"{model_type}_fold_{fold}.keras"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data"))
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--rgb-dir", default=None)
    parser.add_argument("--fft-dir", default=None)
    parser.add_argument("--dual-dir", default=None)
    parser.add_argument("--reports-dir", default=str(BASE_DIR / "reports"))
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    splits_dir = data_root / "splits"
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rgb_index, fft_index = build_processed_indices(data_root, split="train")

    # Preload fold dev & test arrays once
    fold_cache = {}
    for fold in range(1, args.folds + 1):
        dev_file = splits_dir / f"fold_{fold}_dev.txt"
        test_file = splits_dir / f"fold_{fold}_test.txt"
        if dev_file.exists() and test_file.exists():
            fold_cache[fold] = (
                load_split_arrays(read_split(dev_file), rgb_index, fft_index),
                load_split_arrays(read_split(test_file), rgb_index, fft_index),
            )

    model_families = [
        ("rgb", "RGB-Only", args.rgb_dir),
        ("fft", "FFT-Only", args.fft_dir),
        ("dual", "Dual-Stream (RGB+FFT)", args.dual_dir),
    ]

    per_fold_rows = []
    summary_rows = []

    for m_type, m_label, specific_dir in model_families:
        fold_results = []
        total_cm = np.zeros((2, 2), dtype=np.int32)

        for fold in range(1, args.folds + 1):
            m_path = resolve_model_path(m_type, fold, args.models_dir, specific_dir)
            if m_path is None or fold not in fold_cache:
                continue

            if not fold_results:
                print(f"\n=== Evaluasi {m_label} ===")

            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(str(m_path))

            (dev_rgb, dev_fft, dev_y), (test_rgb, test_fft, test_y) = fold_cache[fold]
            dev_scores = predict_scores(model, m_type, dev_rgb, dev_fft)
            threshold = find_eer_threshold(dev_scores, dev_y)

            test_scores = predict_scores(model, m_type, test_rgb, test_fft)
            metrics = compute_all_metrics(test_scores, test_y, threshold)
            metrics["fold"] = fold
            metrics["threshold"] = threshold
            fold_results.append(metrics)

            cm_fold = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
            total_cm += cm_fold

            per_fold_rows.append({
                "model": m_label,
                "fold": fold,
                "threshold": f"{threshold:.4f}",
                "APCER": f"{metrics['apcer']:.4f}",
                "BPCER": f"{metrics['bpcer']:.4f}",
                "ACER": f"{metrics['acer']:.4f}",
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TP": metrics["tp"],
            })

            print(
                f"  Fold {fold}: Thresh={threshold:.3f} | "
                f"APCER={metrics['apcer']*100:.2f}% | BPCER={metrics['bpcer']*100:.2f}% | "
                f"ACER={metrics['acer']*100:.2f}% | Acc={metrics['accuracy']*100:.2f}% | F1={metrics['f1']*100:.2f}%"
            )

        if fold_results:
            apcers = [r["apcer"] for r in fold_results]
            bpcers = [r["bpcer"] for r in fold_results]
            acers = [r["acer"] for r in fold_results]
            accs = [r["accuracy"] for r in fold_results]
            f1s = [r["f1"] for r in fold_results]

            summary_rows.append({
                "Model": m_label,
                "Folds": len(fold_results),
                "APCER (%)": f"{np.mean(apcers)*100:.2f} ± {np.std(apcers)*100:.2f}",
                "BPCER (%)": f"{np.mean(bpcers)*100:.2f} ± {np.std(bpcers)*100:.2f}",
                "ACER (%)": f"{np.mean(acers)*100:.2f} ± {np.std(acers)*100:.2f}",
                "Accuracy (%)": f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
                "F1 (%)": f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
                "TN": int(total_cm[0, 0]),
                "FP": int(total_cm[0, 1]),
                "FN": int(total_cm[1, 0]),
                "TP": int(total_cm[1, 1]),
            })

            cm_path = reports_dir / f"confusion_matrix_{m_type}.png"
            save_confusion_matrix(total_cm, f"Confusion Matrix - {m_label}", cm_path)
            print(f"  -> Confusion matrix disimpan ke: {cm_path}")

    if per_fold_rows:
        per_fold_csv = reports_dir / "evaluation_per_fold.csv"
        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_fold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_fold_rows)

    if summary_rows:
        summary_csv = reports_dir / "evaluation_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n=== RINGKASAN PERBANDINGAN MODEL (MEAN ± STD) ===")
        print(tabulate(summary_rows, headers="keys", tablefmt="grid"))
        print(f"\n✅ Laporan CSV & Confusion Matrix tersimpan di: {reports_dir}")
    else:
        print("\n⚠️ Tidak ada model fold yang ditemukan untuk dievaluasi.")


if __name__ == "__main__":
    main()

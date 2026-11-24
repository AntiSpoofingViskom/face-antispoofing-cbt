import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from model_rgb_only import build_model_rgb
import os
import argparse
import yaml

# --- Konfigurasi ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--data_dir', type=str, default=None)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

DATA_ROOT = args.data_dir if args.data_dir else BASE_DIR

RGB_DIR = os.path.join(DATA_ROOT, config['paths']['processed_rgb'], 'train')
BATCH_SIZE = config['training']['batch_size']
IMG_SIZE = tuple(config['preprocessing']['target_size'])
EPOCHS = config['training']['epochs']
SEED = 123

# --- Load Dataset ---
def load_dataset(directory, subset):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset=subset,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED
    )

print("Memuat dataset RGB...")
rgb_train = load_dataset(RGB_DIR, "training")
rgb_val = load_dataset(RGB_DIR, "validation")

# --- Class Weight ---
def get_class_weights(directory):
    classes = os.listdir(directory)
    counts = []
    for cls in classes:
        count = len(os.listdir(os.path.join(directory, cls)))
        counts.append(count)
    total = sum(counts)
    weights = {i: total / (len(counts) * count) for i, count in enumerate(counts)}
    return weights

print("Menghitung bobot kelas...")
class_weights = get_class_weights(RGB_DIR)
print(f"Bobot Kelas: {class_weights}")

# --- Build & Train ---
model = build_model_rgb(input_shape=(224, 224, 3))

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, 'models', 'antispoofing_rgb_only.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("Memulai training model RGB Only (baseline)...")
history = model.fit(
    rgb_train,
    validation_data=rgb_val,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    class_weight=class_weights
)

# --- Simpan Grafik ---
os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Grafik Akurasi Model RGB Only')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'reports', 'akurasi_rgb_only.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Grafik Loss Model RGB Only')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'reports', 'loss_rgb_only.png'))
plt.close()

print("Training selesai! Model disimpan di models/antispoofing_rgb_only.keras")

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
import os

# --- PERBAIKAN PATH ABSOLUT ---
# Dapatkan lokasi absolut folder utama proyek (naik 1 level dari folder 'src')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Konfigurasi Path dan Parameter (Sambungkan dengan BASE_DIR)
RGB_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'rgb', 'train')
FFT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'fft', 'train')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123

# 2. Memuat Dataset dengan Validation Split
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

print("Memuat dataset FFT...")
fft_train = load_dataset(FFT_DIR, "training")
fft_val = load_dataset(FFT_DIR, "validation")

# PERBAIKAN PENTING: Gunakan format Dictionary untuk input agar tidak salah masuk pintu
def combine_datasets(rgb, fft):
    return tf.data.Dataset.zip((rgb, fft)).map(
        lambda rgb_data, fft_data: ({'rgb_input': rgb_data[0], 'fft_input': fft_data[0]}, rgb_data[1])
    )

train_ds = combine_datasets(rgb_train, fft_train)
val_ds = combine_datasets(rgb_val, fft_val)

# 3. Panggil Model Dual-Stream yang sudah dirapikan dari model.py
model = build_model(rgb_shape=(224, 224, 3), fft_shape=(224, 224, 3))

# 4. Mengatur Checkpoint
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, 'models', 'antispoofing_model.keras'),
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

print("Memulai proses training beserta validasi...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint]
)

# 5. Menyimpan Grafik ke folder reports
os.makedirs('reports', exist_ok=True)

# Plot Akurasi
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('reports/akurasi_plot.png')
plt.close()

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Grafik Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('reports/loss_plot.png')
plt.close()

print("Training selesai! Model terbaik disimpan, dan grafik diekspor ke folder 'reports'.")
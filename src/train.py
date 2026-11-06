import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models
from model import build_model
import os

# 1. Konfigurasi Path dan Parameter (Sudah kamu ubah)
# 1. Konfigurasi Path dan Parameter
RGB_DIR = 'data/processed/rgb/train'
FFT_DIR = 'data/processed/fft/train'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123

# 2. Memuat Dataset dengan Validation Split (YANG DIUBAH)
def load_dataset(directory, subset):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2, # Menyisihkan 20% data untuk validasi
        subset=subset,        # Menentukan 'training' atau 'validation'
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

# Menggabungkan RGB dan FFT
def combine_datasets(rgb, fft):
    return tf.data.Dataset.zip((rgb, fft)).map(
        lambda rgb_data, fft_data: ((rgb_data[0], fft_data[0]), rgb_data[1])
    )

train_ds = combine_datasets(rgb_train, fft_train)
val_ds = combine_datasets(rgb_val, fft_val) # Dataset validasi gabungan

# 3. Membangun Arsitektur Dual-Stream CNN (TETAP SAMA)
input_rgb = layers.Input(shape=(224, 224, 3), name='rgb_input')
x1 = layers.Rescaling(1./255)(input_rgb)
x1 = layers.Conv2D(32, (3, 3), activation='relu')(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Flatten()(x1)

input_fft = layers.Input(shape=(224, 224, 3), name='fft_input')
x2 = layers.Rescaling(1./255)(input_fft)
x2 = layers.Conv2D(32, (3, 3), activation='relu')(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Flatten()(x2)

merged = layers.Concatenate()([x1, x2])
dense1 = layers.Dense(128, activation='relu')(merged)
dense2 = layers.Dropout(0.5)(dense1)
output = layers.Dense(1, activation='sigmoid', name='output')(dense2)

model = build_model(input_shape=(224, 224, 3))

# 4. Kompilasi dan Training Model dengan Checkpoint
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Mengatur checkpoint untuk menyimpan model TERBAIK saja
checkpoint = ModelCheckpoint(
    'models/antispoofing_model.h5', 
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
    callbacks=[checkpoint] # Tambahkan checkpoint di sini
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
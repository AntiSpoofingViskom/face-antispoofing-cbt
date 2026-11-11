import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, Flatten, Rescaling
from tensorflow.keras.models import Model

def build_model(rgb_shape=(224, 224, 3), fft_shape=(224, 224, 3)):
    """Membangun arsitektur model Dual-Input dengan CNN Kustom + Augmentasi Cahaya."""
    
    # 1. Definisikan Input
    rgb_input = Input(shape=rgb_shape, name='rgb_input')
    fft_input = Input(shape=fft_shape, name='fft_input')

    # 2. Alat Augmentasi (Hanya untuk RGB)
    # Factor diturunkan menjadi 0.15 agar tidak terlalu merusak tekstur wajah asli
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(factor=0.15), 
        tf.keras.layers.RandomContrast(factor=0.15),   
    ], name='rgb_augmentation')

    augmented_rgb = data_augmentation(rgb_input)

    # 3. STREAM 1 (RGB): KEMBALI MENGGUNAKAN CNN ORIGINAL-MU
    # Kita tidak lagi memakai MobileNetV3, melainkan CNN yang dilatih dari nol
    x1 = Rescaling(1./255)(augmented_rgb)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Flatten()(x1)

    # 4. STREAM 2 (FFT): CNN ORIGINAL (Tetap sama)
    x2 = Rescaling(1./255)(fft_input)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Flatten()(x2)

    # 5. Gabungkan Pengetahuan dari RGB dan FFT
    merged = Concatenate()([x1, x2])
    
    # 6. Kepala Klasifikasi (Decision Maker)
    dense1 = Dense(128, activation='relu')(merged)
    dense2 = Dropout(0.5)(dense1) # Mencegah hafalan (overfitting)
    outputs = Dense(1, activation='sigmoid', name='output')(dense2)

    # 7. Rangkai dan Compile
    model = Model(inputs=[rgb_input, fft_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
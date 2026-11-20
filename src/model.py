import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Rescaling
from tensorflow.keras.models import Model

def build_model(rgb_shape=(224, 224, 3), fft_shape=(224, 224, 3)):
    """Membangun arsitektur model Dual-Input dengan MobileNetV3Small."""
    
    # 1. Definisikan Input
    rgb_input = Input(shape=rgb_shape, name='rgb_input')
    fft_input = Input(shape=fft_shape, name='fft_input')

    # 2. Augmentasi (Hanya untuk RGB)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(factor=0.15),
        tf.keras.layers.RandomContrast(factor=0.15),
    ], name='rgb_augmentation')

    augmented_rgb = data_augmentation(rgb_input)

    # 3. Normalisasi pixel [0,255] -> [0,1]
    rgb_scaled = Rescaling(1./255)(augmented_rgb)
    fft_scaled = Rescaling(1./255)(fft_input)

    # 4. STREAM 1 (RGB): MobileNetV3Small
    rgb_backbone = MobileNetV3Small(
        input_shape=rgb_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
        name='mobilenet_rgb'
    )
    x1 = rgb_backbone(rgb_scaled)

    # 5. STREAM 2 (FFT): MobileNetV3Small
    fft_backbone = MobileNetV3Small(
        input_shape=fft_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
        name='mobilenet_fft'
    )
    x2 = fft_backbone(fft_scaled)

    # 6. Gabungkan fitur dari kedua stream
    merged = Concatenate()([x1, x2])
    
    # 7. Kepala Klasifikasi
    dense1 = Dense(128, activation='relu')(merged)
    dense2 = Dropout(0.5)(dense1)
    outputs = Dense(1, activation='sigmoid', name='output')(dense2)

    # 8. Rangkai dan Compile
    model = Model(inputs=[rgb_input, fft_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

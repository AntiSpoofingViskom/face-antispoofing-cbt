import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    """Membangun arsitektur model MobileNetV3-Small untuk klasifikasi biner."""
    
    # 1. Panggil "otak dasar" MobileNetV3-Small bawaan TensorFlow
    # include_top=False artinya kita membuang lapisan klasifikasi bawaannya 
    # karena kita hanya ingin menebak 2 kondisi (Asli/Palsu), bukan 1000 objek.
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # 2. Bekukan (freeze) otak dasar ini agar kepintaran awalnya tidak rusak saat dilatih
    base_model.trainable = False 

    # 3. Buat "kepala" baru khusus untuk Anti-Spoofing
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Mencegah overfitting (agar tidak terlalu menghafal)
    
    # 4. Lapisan Output: 1 Neuron (karena biner)
    # Activation 'sigmoid' akan mengeluarkan nilai dari 0.0 sampai 1.0
    # Misal: < 0.5 = Bonafide (Asli), >= 0.5 = Spoof (Palsu)
    outputs = Dense(1, activation='sigmoid')(x)

    # 5. Rangkai tubuh (base) dan kepala (output) menjadi satu model utuh
    model = Model(inputs=base_model.input, outputs=outputs)

    # 6. Atur cara model belajar (Compile)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
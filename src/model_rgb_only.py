import tensorflow as tf
from keras.applications import MobileNetV3Small
from keras.layers import Dense, Dropout, Input
from keras.models import Model

def build_model_rgb(input_shape=(224, 224, 3)):
    """Model baseline single-stream RGB dengan MobileNetV3Small."""

    # 1. Input
    rgb_input = Input(shape=input_shape, name='rgb_input')

    # 2. Augmentasi
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(factor=0.15),
        tf.keras.layers.RandomContrast(factor=0.15),
    ], name='rgb_augmentation')

    augmented_rgb = data_augmentation(rgb_input)

    # 3. Backbone MobileNetV3Small (preprocessing internal handles normalisasi)
    backbone = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
        name='mobilenet_rgb'
    )
    x = backbone(augmented_rgb)

    # 5. Kepala Klasifikasi (sama persis dengan dual-stream)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    # 6. Compile (setting identik)
    model = Model(inputs=rgb_input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

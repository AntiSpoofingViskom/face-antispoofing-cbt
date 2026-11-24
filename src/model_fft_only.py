import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


def build_model_fft(input_shape=(224, 224, 3)):
    """Model single-stream 2D FFT dengan MobileNetV3Small."""
    fft_input = Input(shape=input_shape, name='fft_input')

    backbone = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
        name='mobilenet_fft'
    )
    x = backbone(fft_input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=fft_input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

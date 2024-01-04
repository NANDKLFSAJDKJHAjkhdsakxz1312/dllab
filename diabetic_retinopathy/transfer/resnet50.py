import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)


def build_model(input_shape, num_classes):
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(256, 256, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes=2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

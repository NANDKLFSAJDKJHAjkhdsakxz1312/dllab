import gin
import tensorflow as tf
from .layers import vgg_block,cnn_block

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')

@gin.configurable
def simple_cnn(
    input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate
):
    assert n_blocks > 0, "Number of blocks has to be at least 1."

    inputs = tf.keras.Input(input_shape)
    out = cnn_block(inputs, base_filters)
    for i in range(n_blocks):
        out = cnn_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(
        n_classes, activation="sigmoid" if n_classes == 2 else "softmax"
    )(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="simple_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy" if n_classes == 2 else "categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

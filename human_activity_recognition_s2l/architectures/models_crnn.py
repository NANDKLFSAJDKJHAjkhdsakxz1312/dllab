import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense


def crnn_model(input_shape, num_classes):
    """Build a CRNN model for classification.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (timesteps, features)).
        num_classes (int): Number of classes for the output layer.

    Returns:
        tf.keras.Model: A compiled CRNN model.
    """
    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolution layer
    conv1 = Conv1D(32, kernel_size=3, activation='relu')(input_layer)
    maxpool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(64, kernel_size=3, activation='relu')(maxpool1)
    maxpool2 = MaxPooling1D(pool_size=2)(conv2)

    # LSTM layer
    lstm = Bidirectional(LSTM(64, return_sequences=False))(maxpool2)

    # Dense layer
    dense = Dense(64, activation='relu')(lstm)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dense)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()  # Print the model summary to check the architecture
    return model

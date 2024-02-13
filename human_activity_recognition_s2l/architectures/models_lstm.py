import gin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


@gin.configurable
def lstm_model(num_lstm_units, num_dense_units, dropout_rate, input_shape, num_classes):
    """Builds an LSTM model for classification.

    Args:
        num_lstm_units (int): Number of units in each LSTM layer.
        num_dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for the dropout layers.
        input_shape (tuple): Shape of the input data (e.g., (timesteps, features)).
        num_classes (int): Number of classes for the output layer.

    Returns:
        tf.keras.Model: A compiled LSTM model.
    """
    model = Sequential()
    # Adding LSTM layers with dropout
    model.add(LSTM(num_lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(num_lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(num_lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    # Adding dense layer
    model.add(Dense(num_dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()  # Print the model summary to check the architecture
    return model





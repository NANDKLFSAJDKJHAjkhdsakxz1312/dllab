import gin
import tensorflow as tf
from keras.models import Sequential
import gin
from keras.layers import LSTM, Dense, Dropout

@gin.configurable
def rnn_model(input_shape, num_classes, rnn_units, lstm_units, dense_units, dropout_rate):

    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True)(inputs)
    out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rnn")
    model.summary()

    return model

@gin.configurable
def gru_model(input_shape, num_classes, gru_units, dense_units, dropout_rate):
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.GRU(gru_units, return_sequences=True)(inputs)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_model")
    model.summary()
    return model




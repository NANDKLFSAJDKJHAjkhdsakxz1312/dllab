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

@gin.configurable
def lstm_model(num_recurrent_layers, num_fc_layers, num_hidden_units, dropout_rate, learning_rate,
               stateful, input_shape, num_classes, batch_size=32):
    model = Sequential()

    for i in range(num_recurrent_layers):
        return_sequences = True if i < num_recurrent_layers - 1 else False

        if stateful:
            assert batch_size is not None, "batch_size must be specified for stateful LSTM"
            batch_input_shape = (batch_size,) + input_shape
            model.add(LSTM(num_hidden_units, return_sequences=return_sequences, stateful=stateful,
                           batch_input_shape=batch_input_shape))
        else:
            # 在非状态保持模式下使用原始的input_shape
            model.add(LSTM(num_hidden_units, return_sequences=return_sequences, input_shape=input_shape))

        model.add(Dropout(dropout_rate))

    for _ in range(num_fc_layers):
        model.add(Dense(num_hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))



    return model



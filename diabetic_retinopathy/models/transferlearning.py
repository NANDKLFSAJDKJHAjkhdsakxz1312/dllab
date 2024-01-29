import tensorflow as tf
import gin

@gin.configurable
def resnet50(input_shape, trainable_rate,  n_classes):

    # set the input
    inputs = tf.keras.Input(input_shape)

    # preprocess input data
    preprocessed_input = tf.keras.applications.resnet50.preprocess_input(inputs)

    # build the resnet50 model with transfer learning
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))

    # freeze layers before tuning layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    out = base_model(preprocessed_input)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(0.7)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')
    model.summary()

    return model

@gin.configurable
def densenet201(input_shape, trainable_rate,  n_classes):
    # set the input
    inputs = tf.keras.Input(input_shape)

    # preprocess input data
    preprocessed_input = tf.keras.applications.DenseNet201.preprocess_input(inputs)

    # build the densenet201 model with transfer learning
    base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)

    # fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))

    # freeze layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    out = base_model(preprocessed_input)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(0.7)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='densenet201')
    model.summary()

    return model


import tensorflow as tf
import gin


@gin.configurable
def resnet50(input_shape, trainable_rate,  num_classes):

    # set the input
    inputs = tf.keras.Input(input_shape)

    # preprocess input data
    preprocessed_input = tf.keras.applications.resnet.preprocess_input(inputs)

    # build the resnet50 model with transfer learning
    base_model = tf.keras.applications.resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    # fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))

    # freeze layers before tuning layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # output of model
    out = base_model(preprocessed_input, training=False)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    outputs = tf.keras.layers.Dense(num_classes,activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')
    model.summary()

    return model

@gin.configurable
def densenet121(input_shape, trainable_rate,  num_classes):
    # set the input
    inputs = tf.keras.Input(input_shape)

    # preprocess input data
    preprocessed_input = tf.keras.applications.densenet.preprocess_input(inputs)

    # build the densenet201 model with transfer learning
    base_model = tf.keras.applications.densenet.DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')

    # fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))

    # freeze layers before tuning layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    out = base_model(preprocessed_input, training=False)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(512, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='densenet121')
    model.summary()

    return model


@gin.configurable
def xception(input_shape, trainable_rate,  num_classes):
    # set the input
    inputs = tf.keras.Input(input_shape)

    # preprocess input data
    preprocessed_input = tf.keras.applications.xception.preprocess_input(inputs)

    # build the densenet201 model with transfer learning
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    # fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))

    # freeze layers before tuning layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    out = base_model(preprocessed_input, training=False)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(64, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='xception')
    model.summary()

    return model
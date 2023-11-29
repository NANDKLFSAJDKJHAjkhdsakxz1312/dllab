import gin
import logging
import tensorflow as tf
import os
from diabetic_retinopathy.input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Parse TFRecord function
        def _parse_function(proto):
            keys_to_features = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }
            parsed_features = tf.io.parse_single_example(proto, keys_to_features)
            image = tf.image.decode_jpeg(parsed_features["image"], channels=3)
            image = tf.cast(image, tf.float32)
            label = tf.cast(parsed_features["label"], tf.int32)
            return image, label

        # Define dataset object
        ds_train = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "input_pipeline", "train.tfrecord")]
        ).map(_parse_function)
        ds_val = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "input_pipeline", "val.tfrecord")]
        ).map(_parse_function)
        ds_test = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "input_pipeline", "test.tfrecord")]
        ).map(_parse_function)

        return prepare(ds_train, ds_val, ds_test)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching, n_epochs):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(310 // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(n_epochs)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test

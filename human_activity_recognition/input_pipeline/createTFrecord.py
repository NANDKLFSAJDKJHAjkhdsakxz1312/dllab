import os
import pandas as pd
import numpy as np
import tensorflow  as tf

def write_tf_record_files(window_size, window_shift):
    """build TFRecord file of HAPT dataset"""
    # load data
    data_dir = os.path.dirname(os.path.realpath(__file__))
    train_features_path = os.path.join(data_dir, 'train_features.csv')
    train_labels_path = os.path.join(data_dir, 'train_labels.npy')
    validation_features_path = os.path.join(data_dir, 'validation_features.csv')
    validation_labels_path = os.path.join(data_dir, 'validation_labels.npy')
    test_features_path = os.path.join(data_dir, 'test_features.csv')
    test_labels_path = os.path.join(data_dir, 'test_labels.npy')

    train_features = pd.read_csv(train_features_path).values
    train_labels = np.load(train_labels_path)
    validation_features = pd.read_csv(validation_features_path).values
    validation_labels = np.load(validation_labels_path)
    test_features = pd.read_csv(test_features_path).values
    test_labels = np.load(test_labels_path )

    # sliding window
    ds_train = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    ds_train = ds_train.window(size=window_size, shift=window_shift, drop_remainder=True)
    ds_val = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
    ds_val = ds_val.window(size=window_size, shift=window_shift, drop_remainder=True)
    ds_test = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    ds_test = ds_test.window(size=window_size, shift=window_size, drop_remainder=True)
    ds_train = ds_train.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size,
                                                                                                        drop_remainder=True)
    ds_val = ds_val.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size,
                                                                                                        drop_remainder=True)
    ds_test = ds_test.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size,
                                                                                                        drop_remainder=True)

    # define features
    def _bytes_feature(value):
        """returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def window_example(feature_window, label_window):
        """change the sequence string into an example"""
        feature_window = tf.io.serialize_tensor(feature_window).numpy()
        label_window = tf.io.serialize_tensor(label_window).numpy()
        feature = {'window_sequence': _bytes_feature(feature_window),
                   'label': _bytes_feature(label_window)}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    train_tfrecord_path = os.path.join(data_dir, "train.tfrecord")
    validation_tfrecord_path = os.path.join(data_dir, "validation.tfrecord")
    test_tfrecord_path = os.path.join(data_dir, "test.tfrecord")

    if os.path.exists(train_tfrecord_path) and os.path.exists(validation_tfrecord_path) and os.path.exists(
            test_tfrecord_path):
        print("TFRecord files already exist. Skipping creation.")
        return

    with tf.io.TFRecordWriter(train_tfrecord_path) as writer:
        for feature_window, label_window in ds_train:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())
    with tf.io.TFRecordWriter(validation_tfrecord_path) as writer:
        for feature_window, label_window in ds_val:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())
    with tf.io.TFRecordWriter(test_tfrecord_path) as writer:
        for feature_window, label_window in ds_test:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())

    print("TfRecord files created.")

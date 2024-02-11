import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gin
import logging
from .createTFrecord import write_tf_record_files
import matplotlib.pyplot as plt

@gin.configurable
def load(name, window_size, window_shift, batch_size, buffer_size, drop_remainder):
    """read TFrecord-files and prepare dataset for model"""
    if name == 'HAPT':
        logging.info(f"Preparing dataset {name}...")
        data_dir = os.path.dirname(os.path.realpath(__file__))
        train_tfrecord_path = os.path.join(data_dir, "train.tfrecord")
        validation_tfrecord_path = os.path.join(data_dir, "validation.tfrecord")
        test_tfrecord_path = os.path.join(data_dir, "test.tfrecord")

        write_tf_record_files(window_size=window_size, window_shift=window_shift)

        ds_train = tf.data.TFRecordDataset(train_tfrecord_path)
        ds_val = tf.data.TFRecordDataset(validation_tfrecord_path)
        ds_test = tf.data.TFRecordDataset(test_tfrecord_path)

        # parse data from bytes format into original sequence format
        def _parse_example(window_example):
            keys_to_features = {
                'window_sequence': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.io.parse_single_example(window_example, keys_to_features)
            feature_window = tf.io.parse_tensor(parsed_features['window_sequence'], tf.float64)
            label_window = tf.io.parse_tensor(parsed_features['label'], tf.int32)
            return feature_window, label_window

        ds_train = ds_train.map(_parse_example)
        ds_val = ds_val.map(_parse_example)
        ds_test = ds_test.map(_parse_example)

        # Visualize the original distribution
        class_counts_before = count_classes(ds_train)
        fig = plot_class_distribution(class_counts_before, "Original Distribution")
        fig.savefig("original_distribution.png")

        # prepare the training validation and test datasets
        ds_train = ds_train.shuffle(buffer_size)
        ds_train = ds_train.batch(batch_size, drop_remainder=drop_remainder)
        ds_train = ds_train.repeat(-1)
        ds_val = ds_val.batch(batch_size, drop_remainder=drop_remainder)
        ds_test = ds_test.batch(batch_size, drop_remainder=drop_remainder)

        return ds_train, ds_val, ds_test

def plot_class_distribution(class_counts, title=" "):
    sorted_class_counts = dict(sorted(class_counts.items()))
    labels, counts = zip(*sorted_class_counts.items())
    labels = [str(int(label)) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(labels, counts)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            int(yval),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.show()
    return fig

# count the number of samples per class
def count_classes(ds):
    class_counts = {}
    for _, label in ds.as_numpy_iterator():
        if isinstance(label, np.ndarray):
            for item in label.flat:
                class_counts[item] = class_counts.get(item, 0) + 1
        else:
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts



import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gin
import logging
from .createTFrecord import write_tf_record_files
import matplotlib.pyplot as plt

@gin.configurable
def load(name, window_size, window_shift, batch_size, buffer_size):
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
        feature_description = {
            'window_sequence': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }

        # parse data from bytes format into original sequence format
        def _parse_example(window_example):
            temp = tf.io.parse_single_example(window_example, feature_description)
            feature_window = tf.io.parse_tensor(temp['window_sequence'], tf.float64)
            label_window = tf.io.parse_tensor(temp['label'], tf.int64)
            return feature_window, label_window

        ds_train = ds_train.map(_parse_example)
        ds_val = ds_val.map(_parse_example)
        ds_test = ds_test.map(_parse_example)

        # prepare the training validation and test datasets
        ds_train = ds_train.shuffle(buffer_size)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.repeat(-1)
        ds_val = ds_val.batch(batch_size)
        ds_test = ds_test.batch(batch_size)

        return ds_train, ds_val, ds_test

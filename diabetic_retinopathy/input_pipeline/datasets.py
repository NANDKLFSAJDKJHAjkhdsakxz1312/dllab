import gin
import logging
import tensorflow as tf
import os
from .preprocessing import preprocess, augment
import matplotlib.pyplot as plt

#Show class distribution
def plot_class_distribution(class_counts, title='Class Distribution'):
    labels, counts = zip(*class_counts.items())
    plt.bar(labels, counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.show()

#Count the number of samples per class
def count_classes(ds):
    class_counts = {}
    for _, label in ds.as_numpy_iterator():
        label = label
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts


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
            filenames=[os.path.join(data_dir, "train.tfrecord")]
        ).map(_parse_function)
        ds_val = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "val.tfrecord")]
        ).map(_parse_function)
        ds_test = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "test.tfrecord")]
        ).map(_parse_function)

        return prepare(ds_train, ds_val, ds_test)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching):
    #Visualize the original distribution
    class_counts_before = count_classes(ds_train)
    plot_class_distribution(class_counts_before, 'Original Class Distribution')

    # Resampling
    def class_func(image, label):
        return label

    ds_train_resampled = ds_train.rejection_resample(
        class_func=class_func, target_dist=[0.5, 0.5]
    )

    # Map function to remove class information
    ds_train_resampled = ds_train_resampled.map(lambda extra_label, features_and_label: features_and_label)

    # Visualize the distribution after resampling
    class_counts_after = count_classes(ds_train_resampled)
    plot_class_distribution(class_counts_after, 'Resampled Class Distribution')

    # Prepare training dataset
    ds_train = ds_train_resampled.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
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
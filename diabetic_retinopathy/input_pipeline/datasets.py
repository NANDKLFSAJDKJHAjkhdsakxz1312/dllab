import gin
import logging
import tensorflow as tf
import os
from .preprocessing import preprocess, augment
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from .createTFRecord import prepare_image_paths_and_labels
from .createTFRecord import create_tfrecord
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

@gin.configurable
def load(name, data_dir):
    # choose dataset
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Prepare images and labels
        (
            train_image_paths,
            train_labels,
            val_image_paths,
            val_labels,
            test_image_paths,
            test_labels,
            num_classes,
            label
        ) = prepare_image_paths_and_labels()

        # Create TF files
        create_tfrecord(train_image_paths, train_labels, "train.tfrecord")
        create_tfrecord(val_image_paths, val_labels, "val.tfrecord")
        create_tfrecord(test_image_paths, test_labels, "test.tfrecord")
        print("TfRecord files created.")

        ds_info = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }

        # parse TFRecord
        def _preprocess(proto):
            parsed_features = tf.io.parse_single_example(proto, ds_info)
            image = tf.image.decode_jpeg(parsed_features["image"], channels=3)
            image = tf.cast(image, tf.float32)
            label = tf.cast(parsed_features["label"], tf.int32)
            return image, label

        # Define dataset object
        ds_train = tf.data.TFRecordDataset(filenames=[os.path.join(data_dir, "train.tfrecord")]).map(_preprocess)
        ds_val = tf.data.TFRecordDataset(filenames=[os.path.join(data_dir, "val.tfrecord")]).map(_preprocess)
        ds_test = tf.data.TFRecordDataset(filenames=[os.path.join(data_dir, "test.tfrecord")]).map(_preprocess)

        # Visualize the original distribution
        class_counts_before = count_classes(ds_train)
        fig = plot_class_distribution(class_counts_before, "Original Binary-Class Distribution")
        fig.savefig("original_binary_class_distribution.png")

        # resampling
        def class_func(image, label):
            return label

        ds_train_resampled = ds_train.rejection_resample(class_func=class_func, target_dist=[0.5, 0.5])

        # Map function to remove class information
        ds_train = ds_train_resampled.map(lambda extra_label, features_and_label: features_and_label)

        # Visualize the distribution after resampling
        class_counts_after = count_classes(ds_train)
        fig = plot_class_distribution(class_counts_after, "Resampled Binary_Class Distribution")
        fig.savefig("resampled_Binary_class_distribution.png")

        return prepare(ds_train, ds_val, ds_test, ds_info)

    # use eyespaces dataset
    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")

        train_files, val_files, test_files = get_eyepacs_tfrecord()
        ds_train = tf.data.TFRecordDataset(train_files)
        ds_val = tf.data.TFRecordDataset(val_files)
        ds_test = tf.data.TFRecordDataset(test_files)

        ds_info = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'name': tf.io.FixedLenFeature([], tf.string),
            }

        def _preprocess(proto):
            parsed_features = tf.io.parse_single_example(proto, ds_info)
            image = tf.image.decode_jpeg(parsed_features["image"], channels=3)
            image = tf.image.resize(image, (300, 300))
            label = tf.cast(parsed_features["label"], tf.int32)
            return image, label

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        for image, label in ds_test.take(1):
            image = image.numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(image.astype('uint8'))
            plt.title(f'Label: {label.numpy()}')
            plt.axis('off')  #
            plt.show()
            plt.savefig('btgraham300_example.png',transparent=True, bbox_inches='tight', pad_inches=0)

        class_counts_before = count_classes(ds_train)
        fig = plot_class_distribution(class_counts_before, "Original Multi-Class Distribution")
        fig.savefig("original_multi_lass_distribution.png")

        # resampling
        def class_func(image, label):
            return label

        ds_train_resampled = ds_train.rejection_resample(class_func=class_func, target_dist=[0.2, 0.2, 0.2, 0.2, 0.2])

        # Map function to remove class information
        ds_train = ds_train_resampled.map(lambda extra_label, features_and_label: features_and_label)

        # Visualize the distribution after resampling
        class_counts_after = count_classes(ds_train)
        fig = plot_class_distribution(class_counts_after, "Resampled Multi_Class Distribution")
        fig.savefig("resampled_multi_class_distribution.png")

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, buffer_size):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(buffer_size)
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

    return ds_train, ds_val, ds_test, ds_info

# show class distribution
def plot_class_distribution(class_counts, title=" "):
    sorted_class_counts = dict(sorted(class_counts.items()))
    labels, counts = zip(*sorted_class_counts.items())
    labels = [str(int(label)) for label in labels]
    fig, ax = plt.subplots()
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
    plt.show()
    return fig


# count the number of samples per class
def count_classes(ds):
    class_counts = {}
    for _, label in ds.as_numpy_iterator():
        if isinstance(label, np.ndarray):
            label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def get_eyepacs_tfrecord():
    train_files = []
    val_files = []
    test_files = []
    base_path = '/home/data/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0/'
    for file in os.listdir(base_path):
        full_path = os.path.join(base_path, file)
        if 'train' in file:
            train_files.append(full_path)
        elif 'validation' in file:
            val_files.append(full_path)
        elif 'test' in file:
            test_files.append(full_path)

    return train_files, val_files, test_files



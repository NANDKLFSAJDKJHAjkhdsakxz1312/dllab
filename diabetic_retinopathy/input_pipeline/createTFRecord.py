import tensorflow as tf
from sklearn.utils import shuffle
import os
import pandas as pd
import gin


def _bytes_feature(value):
    # Returns a bytes_list from a string / byte
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # Returns a float_list from a float / double
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(image_paths, labels, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for img_path, label in zip(image_paths, labels):
            image = tf.io.read_file(img_path)
            feature = {
                "image": _bytes_feature(image),
                "label": _int64_feature(label),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


@gin.configurable
def prepare_image_paths_and_labels(data_dir):
    train_labels_df = pd.read_csv(os.path.join(data_dir, "labels", "train.csv"))
    test_labels_df = pd.read_csv(os.path.join(data_dir, "labels", "test.csv"))

    def binary_label(label):
        return 0 if label < 2 else 1

    train_image_paths = [
        os.path.join(data_dir, "images", "train", name + ".jpg")
        for name in train_labels_df["Image name"]
    ]
    test_image_paths = [
        os.path.join(data_dir, "images", "test", name + ".jpg")
        for name in test_labels_df["Image name"]
    ]

    train_labels = train_labels_df["Retinopathy grade"].apply(binary_label).tolist()
    test_labels = test_labels_df["Retinopathy grade"].apply(binary_label).tolist()

    # Shuffle train images
    train_image_paths, train_labels = shuffle(train_image_paths, train_labels)

    # Split train dataset to 6:2:2
    split_idx = int(len(train_image_paths) * 0.75)
    train_image_paths, val_image_paths = (
        train_image_paths[:split_idx],
        train_image_paths[split_idx:],
    )
    train_labels, val_labels = train_labels[:split_idx], train_labels[split_idx:]

    return (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels,
        test_image_paths,
        test_labels,
    )


# Prepare images and labels
data_dir = "D:\idrid\IDRID_dataset"
(
    train_image_paths,
    train_labels,
    val_image_paths,
    val_labels,
    test_image_paths,
    test_labels,
) = prepare_image_paths_and_labels(data_dir)

# Create TF files
create_tfrecord(train_image_paths, train_labels, "train.tfrecord")
create_tfrecord(val_image_paths, val_labels, "val.tfrecord")
create_tfrecord(test_image_paths, test_labels, "test.tfrecord")

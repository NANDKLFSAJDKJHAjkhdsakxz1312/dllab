import gin
import logging
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from diabetic_retinopathy.input_pipeline.preprocessing import preprocess, augment


# Define relationship between images and labels
def create_image_label_map(images_list, labels_df, data_dir):
    label_map = dict(zip(labels_df["Image name"], labels_df["Retinopathy grade"]))

    # Convert to binary question
    def binary_label(label):
        return 0 if label < 2 else 1

    image_paths = []
    labels = []
    for img_path in images_list:
        img_name = os.path.basename(img_path.decode("utf-8"))
        img_name = os.path.splitext(img_name)[0]

        if img_name in label_map:
            label = label_map[img_name]
            full_img_path = os.path.join(data_dir, "images", "train", img_name + ".jpg")
            image_paths.append(full_img_path)
            labels.append(binary_label(label))
        else:
            logging.warning(f"Image name not found in CSV: {img_name}")
    return image_paths, labels


@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # Load CSV files
        train_labels = pd.read_csv(os.path.join(data_dir, "labels", "train.csv"))
        test_labels = pd.read_csv(os.path.join(data_dir, "labels", "test.csv"))

        # load Image files
        train_images = tf.data.Dataset.list_files(
            os.path.join(data_dir, "images", "train", "*.jpg")
        )
        test_images = tf.data.Dataset.list_files(
            os.path.join(data_dir, "images", "test", "*.jpg")
        )

        # Make dataset to list
        train_images_list = list(train_images.as_numpy_iterator())
        test_images_list = list(test_images.as_numpy_iterator())

        # Create pairs of image paths and labels
        train_image_paths, train_labels = create_image_label_map(
            train_images_list, train_labels, data_dir
        )
        test_image_paths, test_labels = create_image_label_map(
            test_images_list, test_labels, data_dir
        )

        # Split the train file to get validation file
        train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
            train_image_paths, train_labels, test_size=103, random_state=7
        )

        # Make images and labels to tf.data.Dataset objects and combine them
        ds_train_images = tf.data.Dataset.from_tensor_slices(train_image_paths)
        ds_train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
        ds_train = tf.data.Dataset.zip((ds_train_images, ds_train_labels))

        ds_val_images = tf.data.Dataset.from_tensor_slices(val_image_paths)
        ds_val_labels = tf.data.Dataset.from_tensor_slices(val_labels)
        ds_val = tf.data.Dataset.zip((ds_val_images, ds_val_labels))

        ds_test_images = tf.data.Dataset.from_tensor_slices(test_image_paths)
        ds_test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
        ds_test = tf.data.Dataset.zip((ds_test_images, ds_test_labels))

        return prepare(ds_train, ds_val, ds_test)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(310 // 10)
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

import gin
import logging
import tensorflow as tf
import os
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, augment

@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # load the image file
        ds_train, ds_val, ds_test = load()
        train_images = tf.data.Dataset.list_files(os.path.join(data_dir, 'train', '*.jpg'))
        test_images = tf.data.Dataset.list_files(os.path.join(data_dir, 'test', '*.jpg'))
        # split the train file to get validation file
        train_images, val_images = train_test_split(train_images, test_size=103, random_state=7)
        # make the file to tf.data.Dataset objects
        ds_train = tf.data.Dataset.from_tensor_slices(train_images).map(lambda x: preprocess(x, label=True),
                                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = tf.data.Dataset.from_tensor_slices(val_images).map(lambda x: preprocess(x, label=True),
                                                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = tf.data.Dataset.from_tensor_slices(test_images).map(lambda x: preprocess(x, label=False),
                                                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds_train, ds_val, ds_test, ds_info

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        #def _preprocess(img_label_dict):
            #return img_label_dict['image'], img_label_dict['label']

        #ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
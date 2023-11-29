import gin
import tensorflow as tf


@gin.configurable
def preprocess(image, label, img_height=256, img_width=256):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.0

    # Resizing image
    image = tf.image.resize(image, [img_height, img_width])

    return image, label


def augment(image, label):
    """Data augmentation"""
    #Randomly rotate the image
    image = tf.image.rot90(
        image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )

    # Randomly flip the image horizontally (left to right).
    image = tf.image.random_flip_left_right(image)

    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.3)

    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label

import tensorflow as tf
import gin


@gin.configurable
def preprocess(image, label, img_height=256, img_width=256):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.0

    # Resizing image
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


def augment(image, label):
    """Data augmentation"""
    choice = tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32)
    # Random contrast adjustment
    #Randomly rotate the image
    if choice == 0:
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Randomly flip the image horizontally (left to right).
    if choice == 1:
        image = tf.image.random_flip_left_right(image)

    # Random brightness adjustment
    if choice == 2:
        image = tf.image.random_brightness(image, max_delta=0.3)

    # Random contrast adjustment
    if choice == 3:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label

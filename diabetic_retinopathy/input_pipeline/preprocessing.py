import gin
import tensorflow as tf

np.set_printoptions(precision=4)


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    return image, label


def augment(image, label):
    """Data augmentation"""

    return image, label

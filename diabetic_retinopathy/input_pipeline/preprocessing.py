import gin
import tensorflow as tf

@gin.configurable
def preprocess(image, label, img_height=256, img_width=256):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.
    #Resizing image
    image = tf.image.resize(image,[img_height, img_width])

    return image, label


def augment(image, label):
    """Data augmentation"""

    return image, label

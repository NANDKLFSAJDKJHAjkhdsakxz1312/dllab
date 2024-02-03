import tensorflow as tf
from PIL import Image
import numpy as np
import io

def augment(image, method):
    if method == 'rotate':
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    elif method == 'flip_lr':
        image = tf.image.flip_left_right(image)
    elif method == 'flip_ud':
        image = tf.image.flip_up_down(image)
    elif method == 'brightness':
        image = tf.image.adjust_brightness(image, 0.2)
    elif method == 'contrast':
        image = tf.image.adjust_contrast(image, 1.2)
    elif method == 'saturation':
        image = tf.image.adjust_saturation(image, 0.7)
    return image


file_path = 'D:/idrid/IDRID_dataset/images/train/IDRiD_099.jpg'
image = tf.io.read_file(file_path)
image = tf.image.decode_jpeg(image, channels=3)

image_resized = tf.image.resize_with_pad(image, 256, 256)
image_resized = tf.cast(image_resized, tf.uint8)

# 准备一个列表来存储增强后的图像
augmented_images = []

methods = ['rotate', 'flip_lr', 'flip_ud', 'brightness', 'contrast', 'saturation']
for method in methods:
    augmented_image = augment(image_resized, method)
    # 将增强后的图像转换为Pillow图像并添加到列表中
    augmented_images.append(Image.fromarray(augmented_image.numpy()))

# 现在您有了一个包含所有增强后图像的列表
# 您可以使用Pillow库来组合这些图像

# 计算组合图像的总宽度
total_width = sum(img.width for img in augmented_images)
max_height = max(img.height for img in augmented_images)

# 创建一个新的空白图像以放置所有增强后的图像
combined_image = Image.new('RGB', (total_width, max_height))

# 拼接图像
x_offset = 0
for img in augmented_images:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width

# 保存组合图像
combined_image.save('D:/idrid/IDRID_dataset/augmented/combined_image.jpg')


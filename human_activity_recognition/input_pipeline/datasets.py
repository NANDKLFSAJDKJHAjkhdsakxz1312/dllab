import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gin
import logging
from .createTFrecord import write_tf_record_files
import matplotlib.pyplot as plt

@gin.configurable
def load(name, window_size, window_shift, batch_size, buffer_size):
    """read TFrecord-files and prepare dataset for model"""
    if name == 'HAPT':
        logging.info(f"Preparing dataset {name}...")
        data_dir = os.path.dirname(os.path.realpath(__file__))
        train_tfrecord_path = os.path.join(data_dir, "train.tfrecord")
        validation_tfrecord_path = os.path.join(data_dir, "validation.tfrecord")
        test_tfrecord_path = os.path.join(data_dir, "test.tfrecord")

        write_tf_record_files(window_size=window_size, window_shift=window_shift)

        ds_train = tf.data.TFRecordDataset(train_tfrecord_path)
        ds_val = tf.data.TFRecordDataset(validation_tfrecord_path)
        ds_test = tf.data.TFRecordDataset(test_tfrecord_path)
        feature_description = {
            'window_sequence': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }

        # parse data from bytes format into original sequence format
        def _parse_example(window_example):
            temp = tf.io.parse_single_example(window_example, feature_description)
            feature_window = tf.io.parse_tensor(temp['window_sequence'], tf.float64)
            label_window = tf.io.parse_tensor(temp['label'], tf.int64)
            return feature_window, label_window

        ds_train = ds_train.map(_parse_example)
        ds_val = ds_val.map(_parse_example)
        ds_test = ds_test.map(_parse_example)

        def plot_data_with_labels(dataset, title, output_dir, color_map):
            """visualizing original data and distribution"""
            features = []
            labels = []
            for feature_batch, label_batch in dataset:
                features.extend(feature_batch.numpy())
                labels.extend(label_batch.numpy())

            # 假设加速度计的x轴数据是特征数组中的第一个元素
            acc_x = np.array(features)[:, :, 0]  # 修改这里以匹配您特征数据的结构

            # 创建一个新的图形和轴对象
            fig, ax = plt.subplots(figsize=(15, 5))

            # 绘制加速度计的x轴数据
            ax.plot(np.arange(len(acc_x)), acc_x, label='acc_x')

            # 为不同的标签添加彩色背景
            start_index = 0
            current_label = labels[0]
            for i, label in enumerate(labels):
                if label != current_label or i == len(labels) - 1:
                    ax.axvspan(start_index, i, color=color_map[current_label], alpha=0.5,
                               label=f'Label {current_label}')
                    start_index = i
                    current_label = label

            # 添加图例和标签
            ax.legend()
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Sensor Reading')

            # 显示图表
            plt.show()





        # prepare the training validation and test datasets
        ds_train = ds_train.shuffle(buffer_size)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.repeat(-1)
        ds_val = ds_val.batch(batch_size)
        ds_test = ds_test.batch(batch_size)

        return ds_train, ds_val, ds_test

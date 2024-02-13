import os
import gin
import pandas as pd
import re
import tensorflow as tf


@gin.configurable
def preprocessor(file_path, window_size, window_step):
    """Preprocess the dataset for window-based analysis.

    Args:
        file_path (str): Path to the directory containing the data files.
        window_size (int): Size of the sliding window.
        window_step (int): Step size of the sliding window.

    Returns:
        tuple: Contains dictionaries of labeled windows and windowed datasets.
    """
    data_folder = file_path

    # Create a dictionary to hold datasets
    combined_datasets = {"Test": {}, "Train": {}, "Validation": {}}

    # Iterate over files and load data
    for file_name in os.listdir(data_folder):
        match = re.match(r'(acc|gyro)_exp(\d+)_user(\d+)\.txt', file_name)
        if not match:
            continue

        sensor_type = match.group(1)  # 'acc' or 'gyro'
        experiment_number = int(match.group(2))
        user_number = int(match.group(3))

        data = pd.read_csv(os.path.join(data_folder, file_name), delimiter=" ", header=None)

        if 22 <= user_number <= 27:
            dataset_name = "Test"
        elif 1 <= user_number <= 21:
            dataset_name = "Train"
        elif 28 <= user_number <= 30:
            dataset_name = "Validation"
        else:
            continue

        experiment_key = (experiment_number, user_number)

        if experiment_key not in combined_datasets[dataset_name]:
            combined_datasets[dataset_name][experiment_key] = {"acc": pd.DataFrame(), "gyro": pd.DataFrame()}

        combined_datasets[dataset_name][experiment_key][sensor_type] = data

    # Merge accelerometer and gyroscope data
    for dataset_name, dataset in combined_datasets.items():
        for key, sensors_data in dataset.items():
            acc_data = sensors_data["acc"]
            gyro_data = sensors_data["gyro"]
            if not acc_data.empty and not gyro_data.empty:
                combined_data = pd.concat([acc_data, gyro_data], axis=1)
                dataset[key] = combined_data
            else:
                del dataset[key]  # Remove entries missing sensor data

    def normalize_data(data):
        """Normalize the data."""
        mean = data.mean()
        std = data.std()
        return (data - mean) / std

    # Normalize data for each experiment in each dataset
    for dataset_name, dataset in combined_datasets.items():
        for key, value in dataset.items():
            dataset[key] = normalize_data(value)

    def sliding_window(data, size, step, experiment_number, user_number):
        """Apply sliding window technique and keep track of experiment and user number."""
        num_windows = ((data.shape[0] - size) // step) + 1
        for i in range(num_windows):
            start = i * step
            end = start + size
            window_data = data[start:end]
            yield window_data, experiment_number, user_number

    # Apply sliding window to each experiment in each dataset
    windowed_datasets = {"Test": [], "Train": [], "Validation": []}
    for dataset_name, dataset in combined_datasets.items():
        for (experiment_number, user_number), data in dataset.items():
            for window_data, exp_num, user_num in sliding_window(data, window_size, window_step, experiment_number,
                                                                 user_number):
                windowed_datasets[dataset_name].append((window_data, exp_num, user_num))

    labels_df = pd.read_csv(os.path.join(data_folder, 'labels.txt'), header=None, delimiter=' ')
    labels_df.columns = ['Experiment', 'User', 'Activity', 'Start', 'End']

    # Adjust label values from 1-12 to 0-11
    labels_df['Activity'] = labels_df['Activity'] - 1

    def label_for_window(window_start, window_end, labels_df, current_experiment, current_user):
        """Determine the label for a window."""
        # Select records for the current experiment, current user, and overlap with the activity time window
        overlapping_labels = labels_df[(labels_df['Experiment'] == current_experiment) &
                                       (labels_df['User'] == current_user) &
                                       (labels_df['Start'] <= window_end) &
                                       (labels_df['End'] >= window_start)].copy()

        if not overlapping_labels.empty:
            # Calculate the number of overlapping samples for each activity
            overlapping_labels['OverlapCount'] = overlapping_labels.apply(
                lambda row: min(window_end, row['End']) - max(window_start, row['Start']), axis=1)

            # Sum the total overlapping samples for each activity
            activity_counts = overlapping_labels.groupby('Activity')['OverlapCount'].sum()

            # Find the activity label with the maximum count of samples
            max_count_activity = activity_counts.idxmax()
            return max_count_activity
        else:
            return None

    # Assign labels to each window
    labeled_windows = {"Test": [], "Train": [], "Validation": []}

    for dataset_name, dataset_windows in windowed_datasets.items():
        for window_data, experiment_number, user_number in dataset_windows:
            window_start = window_data.index[0]
            window_end = window_data.index[-1]
            label = label_for_window(window_start, window_end, labels_df, experiment_number, user_number)
            if label is not None:
                labeled_windows[dataset_name].append((window_data, label, experiment_number, user_number))

    # Initialize a dictionary to store the count of each label
    label_counts = {}

    # Iterate through all windows in the Test dataset
    for _, label, _, _ in labeled_windows["Test"]:
        if label in label_counts:
            # If the label already exists, increment its count
            label_counts[label] += 1
        else:
            # If it's a new label, initialize its count to 1
            label_counts[label] = 1

    # Function to convert labeled windows into TensorFlow datasets
    def convert_to_tensor_dataset(labeled_windows):
        datasets = {}
        for dataset_name, windows_with_labels in labeled_windows.items():
            # Convert window data and labels into tensors
            window_data_tensors = [tf.convert_to_tensor(window_data.values, dtype=tf.float32) for
                                   window_data, label, _, _ in windows_with_labels]
            labels_tensors = [tf.convert_to_tensor(label, dtype=tf.int32) for window_data, label, _, _ in
                              windows_with_labels]

            # Create TensorFlow Dataset
            dataset = tf.data.Dataset.from_tensor_slices((window_data_tensors, labels_tensors))
            datasets[dataset_name] = dataset
        return datasets

    tensor_datasets = convert_to_tensor_dataset(labeled_windows)

    # Serialization function for TFRecord
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(window_data, label):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature = {
            'window_data': _bytes_feature(tf.io.serialize_tensor(window_data)),
            'label': _bytes_feature(tf.io.serialize_tensor(label))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # Write serialized data to TFRecord files
    for dataset_name, dataset in tensor_datasets.items():
        # Relative path
        tfrecord_file_path = os.path.join('input_pipeline_s2l', f'{dataset_name}.tfrecords')
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            for window_data, label in dataset:
                example = serialize_example(window_data, label)
                writer.write(example)

    return labeled_windows, windowed_datasets

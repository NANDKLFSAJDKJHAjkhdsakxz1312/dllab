from input_pipeline_s2l.preprocessing import preprocessor
from architectures.models_lstm import lstm_model
import gin
from input_pipeline_s2l.datasets import load
import tensorflow as tf
from utils import utils_params
from train import Trainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# use lstm model trained to do visualization
run_paths = utils_params.gen_run_folder('lstm')
checkpoint_paths = run_paths["path_ckpts_train"]

# gin-config
gin.parse_config_files_and_bindings(['configs/config.gin'], [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())

labeled_windows, windowed_datasets = preprocessor()

_, _, ds_test = load()

model = lstm_model(input_shape=(250, 6), num_classes=12)

# Find the latest checkpoint file
latest_ckpt = tf.train.latest_checkpoint(checkpoint_paths)
if latest_ckpt:
    ckpt = tf.train.Checkpoint(
        net=model
    )
    ckpt.restore(latest_ckpt)
    print(f"Restored from {latest_ckpt}")
else:
    print("No checkpoint found.")

# Your windowed_datasets data
windowed_datasets = windowed_datasets

# Select specific user and experiment
selected_user = 27
selected_experiment = 55

# Filter out all windows for a specific user and experiment
selected_windows = [window for window, experiment, user in windowed_datasets['Test'] if user == selected_user and experiment == selected_experiment]

# Initialize total datasets for acceleration and gyroscope
total_acc_data = pd.DataFrame()
total_gyro_data = pd.DataFrame()

window_step = gin.query_parameter('preprocessor.window_step')

# Merge data for each window
for i, window in enumerate(selected_windows):
    # Assume the first 3 columns are acceleration data and the last 3 columns are gyroscope data
    acc_data = window.iloc[:, 0:3]
    gyro_data = window.iloc[:, 3:6]

    # Merge data into total datasets, considering the window step
    total_acc_data = pd.concat([total_acc_data, acc_data.iloc[-window_step:]], ignore_index=True)
    total_gyro_data = pd.concat([total_gyro_data, gyro_data.iloc[-window_step:]], ignore_index=True)

# Plot acceleration curves
plt.figure(figsize=(10, 4))
plt.plot(total_acc_data)
plt.title('Acceleration Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.legend(['X', 'Y', 'Z'])
plt.savefig('acceleration_data_plot.png')  # Save the plot


# Plot gyroscope curves
plt.figure(figsize=(10, 4))
plt.plot(total_gyro_data)
plt.title('Gyroscope Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.legend(['X', 'Y', 'Z'])
plt.savefig('gyroscope_data_plot.png')  # Save the plot


labeled_windows = labeled_windows  # Your labeled_windows data

# Extract window index intervals and labels for the selected user and experiment
window_intervals_labels = [(window.index[0], window.index[-1], label) for window, label, experiment, user in labeled_windows['Test'] if user == selected_user and experiment == selected_experiment]

activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
    6: "STAND_TO_SIT",
    7: "SIT_TO_STAND",
    8: "SIT_TO_LIE",
    9: "LIE_TO_SIT",
    10: "STAND_TO_LIE",
    11: "LIE_TO_STAND"
}

# Assign colors to each activity
activity_colors = {
    0: 'lightcoral',
    1: 'lightgreen',
    2: 'lightskyblue',
    3: 'paleturquoise',
    4: 'orchid',
    5: 'lightyellow',
    6: 'moccasin',
    7: 'plum',
    8: 'burlywood',
    9: 'pink',
    10: 'lightgrey',
    11: 'khaki'
}

# Create a figure
plt.figure(figsize=(10, 8))

# Create subplots for acceleration and gyroscope data
ax1 = plt.subplot(2, 1, 1)
for start, end, label in window_intervals_labels:
    ax1.axvspan(start, end, color=activity_colors[label])
ax1.plot(total_acc_data)
ax1.set_title('Ground Truth: Acceleration Data for Selected User and Experiment')
ax1.set_ylabel('Acceleration')


ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for start, end, label in window_intervals_labels:
    ax2.axvspan(start, end, color=activity_colors[label])
ax2.plot(total_gyro_data)
ax2.set_title('Ground Truth: Gyroscope Data for Selected User and Experiment')
ax2.set_xlabel('Index')
ax2.set_ylabel('Angular Velocity')

# Save the plot
plt.savefig('ground_truth_gyroscope_plot.png')

# Create a legend
legend_elements = [Patch(facecolor=activity_colors[label], label=activity_labels[label]) for label in activity_colors]
plt.legend(handles=legend_elements, loc='upper center', ncol=6, bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.show()

# Set window step, window size, and batch size
window_step = gin.query_parameter('preprocessor.window_step')
window_size = gin.query_parameter('preprocessor.window_size')
batch_size = gin.query_parameter('prepare.batch_size')

# Convert window data in selected_windows to TensorFlow tensors
window_data_tensors = [tf.convert_to_tensor(window.values, dtype=tf.float32) for window in selected_windows]

# Create TensorFlow Dataset
ds_selected = tf.data.Dataset.from_tensor_slices(window_data_tensors)
ds_selected = ds_selected.batch(batch_size, drop_remainder=True)
predicted_labels_with_indices = []
current_index = 0
batch_data_accumulated = []
total_windows_count = 0

for batch in ds_selected:
    batch_data = batch
    current_batch_size = len(batch_data)
    for i in range(current_batch_size):
        original_start_index = current_index + i * window_step
        original_end_index = original_start_index + window_size
        window_data = batch_data[i].numpy()
        batch_data_accumulated.append(window_data)

        if len(batch_data_accumulated) == batch_size:
            batch_data_np = np.array(batch_data_accumulated)
            predicted_labels_batch = model.predict(batch_data_np)
            predicted_labels = np.argmax(predicted_labels_batch, axis=1)

            for j, predicted_label in enumerate(predicted_labels):
                start_index = original_start_index + j * window_step - (batch_size - 1) * window_step
                end_index = start_index + window_size
                print(f"Window Index: Start {start_index}, End {end_index}, Label {predicted_label}")
                predicted_labels_with_indices.append((start_index, end_index, predicted_label))

            batch_data_accumulated = []

    current_index += current_batch_size * window_step

# Print total window count and length of predicted_labels_with_indices
print("Total Window Count:", total_windows_count)
print("Length of predicted_labels_with_indices list:", len(predicted_labels_with_indices))


# Create a subplot layout with 2 rows and 1 column, sharing the x-axis
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot acceleration data
for start_index, end_index, label in predicted_labels_with_indices:
    axs[0].axvspan(start_index, end_index, color=activity_colors[label])
axs[0].plot(total_acc_data)
axs[0].set_title('Prediction: Acceleration Data for Selected User and Experiment')
axs[0].set_ylabel('Acceleration')


# Plot gyroscope data
for start_index, end_index, label in predicted_labels_with_indices:
    axs[1].axvspan(start_index, end_index, color=activity_colors[label])
axs[1].plot(total_gyro_data)
axs[1].set_title('Prediction: Gyroscope Data for Selected User and Experiment')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Angular Velocity')

# Save the plot
plt.savefig('predicted_gyroscope_plot.png')


legend_elements = [Patch(facecolor=activity_colors[label], label=activity_labels[label]) for label in activity_colors]
plt.legend(handles=legend_elements, loc='upper center', ncol=6, bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.show()

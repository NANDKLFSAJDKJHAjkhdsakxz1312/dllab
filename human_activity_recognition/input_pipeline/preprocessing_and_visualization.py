import gin
import pandas as pd
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

features = {'train':[], 'validation':[], 'test':[]}
labels = {'train':[], 'validation':[], 'test':[]}

def preprocess(file_name, data_dir, labels_file, output_dir):
    global features, labels
    # read accelerometer files and gyroscope files
    acc_file = os.path.join(data_dir, f'acc_{file_name}.txt')
    gyro_file = os.path.join(data_dir, f'gyro_{file_name}.txt')

    if not os.path.exists(acc_file) or not os.path.exists(gyro_file):
        return
    os.makedirs(output_dir, exist_ok=True)

    # read data in each file
    acc_data = pd.read_csv(acc_file, header=None, sep=' ', names=["acc_x","acc_y","acc_z"])
    gyro_data = pd.read_csv(gyro_file, header=None, sep=' ',names=["gyro_x","gyro_y","gyro_z"])

    # normalize data using Z-score
    normalized_acc = acc_data.apply(stats.zscore)
    normalized_gyro = gyro_data.apply(stats.zscore)

    # combine accelerometer files and gyroscope files
    combined_data = pd.concat([normalized_acc, normalized_gyro], axis=1)

    # read labels and seperate them
    oringinal_label = pd.read_csv(labels_file, header=None, sep=' ')
    separated_label = {exp_id: group for exp_id, group in oringinal_label.groupby(0)}

    exp_id = int(file_name.split('_')[0][3:])
    sequence_labels = np.full(len(combined_data), -1)
    current_labels = separated_label.get(exp_id)
    for _, label_row in current_labels.iterrows():
        start_index = int(label_row[3])
        end_index = int(label_row[4])
        label_value = int(label_row[2])
        sequence_labels[start_index:end_index + 1] = label_value

    # create a boolean index array, marking all labels that are not -1
    valid_indices = sequence_labels != -1
    # change labels from 1-12 to 0-11
    sequence_labels = sequence_labels[valid_indices] - 1
    combined_data = combined_data[valid_indices]

    # define train, validation and test experiment ranges
    train_exp_id = range(1, 44)
    val_exp_id = range(56, 62)
    test_exp_id = range(44, 56)

    dataset_type = ''
    if exp_id in train_exp_id:
        dataset_type = 'train'
    elif exp_id in val_exp_id:
        dataset_type = 'validation'
    elif exp_id in test_exp_id:
        dataset_type = 'test'
    else:
        raise ValueError("Unknown experiment ID.")

    features[dataset_type].append(combined_data)
    labels[dataset_type].append(sequence_labels)

def save_combined_data(output_dir, dataset_type):
    combined_features = pd.concat(features[dataset_type])
    combined_labels = np.concatenate(labels[dataset_type])
    combined_features.to_csv(os.path.join(output_dir, f'{dataset_type}_features.csv'), index=False)
    np.save(os.path.join(output_dir, f'{dataset_type}_labels.npy'), combined_labels)

def plot_sensor_data(data_file, title, output_dir):
    data = pd.read_csv(data_file)
    data = data.iloc[:13000]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.plot(data['acc_x'], label='acc_x')
    ax1.plot(data['acc_y'], label='acc_y')
    ax1.plot(data['acc_z'], label='acc_z')
    ax1.set_title('Accelerometer Data of ' + title)
    ax1.set_ylabel('Accelerometer Value')
    ax1.legend()

    ax2.plot(data['gyro_x'], label='gyro_x')
    ax2.plot(data['gyro_y'], label='gyro_y')
    ax2.plot(data['gyro_z'], label='gyro_z')
    ax2.set_title('Gyroscope Data of ' + title)
    ax2.set_ylabel('Gyroscope Value')
    ax2.legend()

    plt.tight_layout()

    output_file_path = os.path.join(output_dir, title.replace(' ', '_') + '.png')
    plt.savefig(output_file_path)
    print(f"Image saved to {output_file_path}")
    plt.show()

def plot_sensor_data_with_labels(data_file, label_file, title, output_dir):
    data = pd.read_csv(data_file)
    labels = np.load(label_file)

    data = data.iloc[:13000]
    labels = labels[:13000]

    unique_labels = np.unique(labels)
    color_map = plt.get_cmap('Set3')
    num_of_colors = len(unique_labels)
    colors = color_map(np.linspace(0, 1, num_of_colors))
    label_color_map = dict(zip(unique_labels, colors))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.plot(data['acc_x'], label='acc_x')
    ax1.plot(data['acc_y'], label='acc_y')
    ax1.plot(data['acc_z'], label='acc_z')
    ax2.plot(data['gyro_x'], label='gyro_x')
    ax2.plot(data['gyro_y'], label='gyro_y')
    ax2.plot(data['gyro_z'], label='gyro_z')

    ax1.set_title('Accelerometer Data of ' + title)
    ax1.set_ylabel('Accelerometer Value')
    ax1.legend()
    ax2.set_title('Gyroscope Data of ' + title)
    ax2.set_ylabel('Gyroscope Value')
    ax2.legend()

    for i, label in enumerate(labels):
        color = label_color_map.get(label, 'white')
        ax1.axvspan(i, i + 1, color=color, alpha=0.5)
        ax2.axvspan(i, i + 1, color=color, alpha=0.5)

    label_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
                   'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

    # create legend
    legend_patches = [mpatches.Patch(color=color_map(i / num_of_colors), label=label) for i, label in
                      enumerate(label_names)]

    plt.figlegend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=6, labelspacing=0.)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    output_file_path = os.path.join(output_dir, title.replace(' ', '_') + '.png')
    plt.savefig(output_file_path)
    print(f"Image saved to {output_file_path}")
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = 'D:/HAPT/RawData'
    labels_file = 'D:/HAPT/RawData/labels.txt'
    output_dir = os.path.join(current_dir)
    num_experiments = 61
    num_users = 30

    experiments_users = [f'exp{exp_id:02d}_user{user_id:02d}'
                         for exp_id in range(1, num_experiments + 1)
                         for user_id in range(1, num_users + 1)]

    for file_name in experiments_users:
        preprocess(file_name, data_dir,labels_file, output_dir)

    save_combined_data(output_dir, 'train')
    save_combined_data(output_dir, 'test')
    save_combined_data(output_dir, 'validation')

    plot_sensor_data(os.path.join(output_dir, 'train_features.csv'), 'train', output_dir)
    plot_sensor_data(os.path.join(output_dir, 'test_features.csv'), 'test', output_dir)
    plot_sensor_data(os.path.join(output_dir, 'validation_features.csv'), 'validation', output_dir)

    train_file = os.path.join(output_dir, 'train_features.csv')
    train_label = os.path.join(output_dir, 'train_labels.npy')
    plot_sensor_data_with_labels(train_file, train_label, 'Train with labels', output_dir)

    validation_file = os.path.join(output_dir, 'validation_features.csv')
    validation_label = os.path.join(output_dir, 'validation_labels.npy')
    plot_sensor_data_with_labels(validation_file, validation_label, 'Validation with labels', output_dir)

    test_file = os.path.join(output_dir, 'test_features.csv')
    test_label = os.path.join(output_dir, 'test_labels.npy')
    plot_sensor_data_with_labels(test_file, test_label, 'Test with labels', output_dir)












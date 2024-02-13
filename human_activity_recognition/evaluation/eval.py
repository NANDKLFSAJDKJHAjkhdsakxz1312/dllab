import tensorflow as tf
from .metrics import ConfusionMatrix
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def evaluate(model, ds_test, checkpoint_paths):
    # Load Checkpoints
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_paths, max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    step = int(checkpoint.step.numpy())

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    num_classes = 12
    label = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING", "STAND_TO_SIT",
                   "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    test_confusion_matrix = ConfusionMatrix(num_classes, label)

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
            tf.reshape(labels, [-1]), tf.reshape(predictions, [-1, num_classes])
        )
        test_loss(t_loss)
        test_accuracy(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1, num_classes]))
        test_confusion_matrix.update_state(tf.reshape(labels, [-1]),
                                           tf.argmax(tf.reshape(predictions, [-1, num_classes]), axis=1))

    # Iterate over the test dataset
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)
        step += 1
        wandb.log(
            {
                "test_acc": test_accuracy.result(),
                "test_loss": test_loss.result(),
                "step": step,
            }
        )

    # Log and return the test metricsa
    print(
        f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}"
    )

    test_confusion_matrix.summary()
    test_confusion_matrix.plot()

    return test_loss.result(), test_accuracy.result()

def visualization(model_name, model, ds_test, checkpoint_paths):
    # Load Checkpoints
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_paths, max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    predictions_list = []
    for test_images, test_labels in ds_test:
        predictions = model(test_images, training=False)
        predicted_labels = tf.argmax(predictions, axis=-1).numpy()
        predictions_list.append(predicted_labels)

    all_predictions = np.concatenate(predictions_list).flatten()
    np.save(f'model_predictions_{model_name}.npy', all_predictions)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(current_dir)
    project_base_dir = os.path.dirname(os.path.dirname(current_dir))
    predict_file = os.path.join(project_base_dir, 'human_activity_recognition','input_pipeline', 'test_features.csv')
    predict_label = f'model_predictions_{model_name}.npy'
    plot_sensor_data_with_labels(predict_file, predict_label, f'{model_name} prediction with labels', output_dir)

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





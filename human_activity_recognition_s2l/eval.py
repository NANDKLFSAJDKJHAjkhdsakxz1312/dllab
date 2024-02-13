import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix using seaborn.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list of str): List of class names.
    """
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def evaluate(model, checkpoint_dir, ds_test):
    """Evaluate the model using the test dataset and print balanced accuracy.

    Args:
        model: TensorFlow model to be evaluated.
        checkpoint_dir (str): Directory where the checkpoints are stored.
        ds_test (tf.data.Dataset): Test dataset.

    Returns:
        float: Balanced accuracy.
        np.array: Confusion matrix.
    """
    # Checkpoint path - Find the latest checkpoint file
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
        ckpt.restore(latest_ckpt)
        print(f"Restored from {latest_ckpt}")
    else:
        print("No checkpoint found.")

    # Get the number of classes
    num_classes = model.output_shape[-1]

    # Initialize confusion matrix
    confusion_mtx = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        return labels, tf.argmax(predictions, axis=1)

    # Iterate over the test dataset
    for test_images, test_labels in ds_test:
        labels, preds = test_step(test_images, test_labels)
        # Update confusion matrix
        for i in range(len(labels)):
            confusion_mtx[labels[i]][preds[i]] += 1

    # Calculate recall for each class
    recalls = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    # Calculate balanced accuracy
    balanced_accuracy = np.nanmean(recalls) * 100  # Use nanmean to avoid NaN values if a class does not appear

    # Print results
    print(f"Balanced Accuracy: {balanced_accuracy}%")

    # List of class names, replace with actual class names
    class_names = [
        "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6",
        "Class 7", "Class 8", "Class 9", "Class 10", "Class 11", "Class 12"
    ]

    # Plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, class_names)

    return balanced_accuracy, confusion_mtx

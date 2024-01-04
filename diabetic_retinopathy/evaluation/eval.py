import tensorflow as tf
from .metrics import ConfusionMatrix
import os


def count_classes(ds):
    class_counts = {}
    for images, labels in ds:
        for label in labels.numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def evaluate(model, ds_test, checkpoint_paths, num_classes, label):
    # # Load Checkpoints
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001)
    # ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # latest_ckpt = tf.train.latest_checkpoint(checkpoint_paths)
    # if latest_ckpt:
    #     ckpt.restore(ckpt).expect_partial()
    #     print(f"Restored from {latest_ckpt}")
    # model.compile(
    #     optimizer=optimizer,
    #     loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy",
    #     metrics=["accuracy"]
    # )

    # Load Models
    saved_model_path = os.path.join(checkpoint_paths, "saved_model")
    model = tf.keras.models.load_model(saved_model_path)

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    test_confusion_matrix = ConfusionMatrix(num_classes, label)

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            labels, predictions
        )
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        test_confusion_matrix.update_state(labels, tf.argmax(predictions, axis=1))

    # Iterate over the test dataset
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    class_counts = count_classes(ds_test)
    print("Class Distribution in Test Dataset:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")

    # Log and return the test metrics
    print(
        f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}"
    )

    test_confusion_matrix.summary()
    test_confusion_matrix.plot()

    return test_loss.result(), test_accuracy.result()

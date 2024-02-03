import tensorflow as tf
from .metrics import ConfusionMatrix
import os


def evaluate(model, ds_test, checkpoint_paths):
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

    # Log and return the test metrics
    print(
        f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}"
    )

    test_confusion_matrix.summary()
    test_confusion_matrix.plot()

    return test_loss.result(), test_accuracy.result()

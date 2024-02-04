import tensorflow as tf
from .metrics import ConfusionMatrix
import os
import wandb
import numpy as np

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

    test_labels_list = []

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





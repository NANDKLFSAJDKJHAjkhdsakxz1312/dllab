import tensorflow as tf
from .metrics import ConfusionMatrix
import wandb


def evaluate(model, ds_test, checkpoint_paths, num_classes, labels):
    # Load Checkpoints
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),  net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_paths, max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    step = int(checkpoint.step.numpy())

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    test_confusion_matrix = ConfusionMatrix(num_classes, labels)

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        test_confusion_matrix.update_state(labels, tf.argmax(predictions, axis=1))

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

    # Log and return the test metrics
    print(
        f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}"
    )

    test_confusion_matrix.summary()
    test_confusion_matrix.plot()

    return test_loss.result(), test_accuracy.result()

def evaluate_regression(model, ds_test, checkpoint_paths):
    # Load Checkpoints
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_paths, max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    step = int(checkpoint.step.numpy())

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.Huber(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_mae = tf.keras.metrics.MeanAbsoluteError(name="test_mae")
    test_mse = tf.keras.metrics.MeanSquaredError(name="test_mse")

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.Huber()(labels, predictions)
        test_loss(t_loss)
        test_mae(labels, predictions)
        test_mse(labels, predictions)

    # Iterate over the test dataset
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    log_data = {
        "test_loss": test_loss.result().numpy(),
        "test_mae": test_mae.result().numpy(),
        "test_mse": test_mse.result().numpy()
    }
    wandb.log(log_data)

    results = {
        "Test Loss": test_loss.result().numpy(),
        "Test MAE": test_mae.result().numpy(),
        "Test MSE": test_mse.result().numpy()
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    return results

    return results



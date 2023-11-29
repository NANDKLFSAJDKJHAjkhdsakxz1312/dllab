import tensorflow as tf
from diabetic_retinopathy.train import Trainer


def evaluate(model, ds_test, run_paths):
    # Load Checkpoints
    checkpoint_path = run_paths["path_ckpts_train"]
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    if latest_ckpt:
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(latest_ckpt).expect_partial()

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            labels, predictions
        )
        test_loss(t_loss)
        test_accuracy(labels, predictions)

        # Iterate over the test dataset

    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    # Log and return the test metrics
    print(
        f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}"
    )

    return

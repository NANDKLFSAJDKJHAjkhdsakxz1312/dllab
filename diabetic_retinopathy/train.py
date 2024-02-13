import gin
import tensorflow as tf
import logging
import os
import wandb


@gin.configurable
class Trainer(object):
    def __init__(
        self,
        model,
        ds_train,
        ds_val,
        run_paths,
        total_steps,
        log_interval,
        ckpt_interval,
    ):
        self.model = model
        self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Summary Writer
        self.run_paths = run_paths
        self.summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_train"])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10
        )

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

        # set profiling
        self.profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
        )
        self.profiler_summary_path = self.run_paths["path_summary_profiler"]

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        # set the profiler (optional)
        # options = self.profiler_options
        # tf.profiler.experimental.start(self.profiler_summary_path, options=options)

        # Continue training from a saved checkpoint
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        latest_ckpt = tf.train.latest_checkpoint(self.run_paths["path_ckpts_train"])
        if latest_ckpt:
            tf.print(f"Restored from {latest_ckpt}")
        else:
            tf.print("Initializing from scratch.")

        for idx, (images, labels) in enumerate(self.ds_train):
            print(f"Training step {idx+1}")
            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                info = (
                    f"Step {step}, "
                    f"Loss: {self.train_loss.result():.4f}, "
                    f"Accuracy: {self.train_accuracy.result() * 100:.2f}%, "
                    f"Validation Loss: {self.val_loss.result():.4f}, "
                    f"Validation Accuracy: {self.val_accuracy.result() * 100:.2f}%"
                )
                print(info)

                template = "Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}"
                logging.info(
                    template.format(
                        step,
                        self.train_loss.result(),
                        self.train_accuracy.result() * 100,
                        self.val_loss.result(),
                        self.val_accuracy.result() * 100,
                    )
                )

                # wandb logging
                wandb.log(
                    {
                        "train_acc": self.train_accuracy.result() * 100,
                        "train_loss": self.train_loss.result(),
                        "val_acc": self.val_accuracy.result() * 100,
                        "val_loss": self.val_loss.result(),
                        "step": step,
                    }
                )

                # Write summary to tensorboard
                with self.summary_writer.as_default():
                    tf.summary.scalar("Loss", self.train_loss.result(), step=step)
                    tf.summary.scalar("Accuracy", self.train_accuracy.result(), step=step)
                    tf.summary.scalar("Validation Loss", self.val_loss.result(), step=step)
                    tf.summary.scalar("Validation Accuracy", self.val_accuracy.result(), step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(
                    f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.'
                )
                # Save checkpoint
                self.ckpt_manager.save()

            # finish
            if step % self.total_steps == 0:
                logging.info(f"Finished training after {step} steps.")
                # Save final checkpoint
                self.ckpt_manager.save()

                # stop the profiler
                # tf.profiler.experimental.stop()

                return self.val_accuracy.result().numpy()

    def save_model(self):
        save_path = os.path.join(self.run_paths["path_ckpts_train"], "saved_model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

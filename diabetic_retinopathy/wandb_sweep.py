import logging
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import vgg_like
from models.architectures import simple_cnn
from train import Trainer
from utils import utils_params, utils_misc
from input_pipeline.createTFRecord import create_tfrecord
from input_pipeline.createTFRecord import prepare_image_paths_and_labels
import os


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f"{key}={value}")

        # generate folder structures
        run_paths = utils_params.gen_run_folder(",".join(bindings))

        # set loggers
        log_file_path = os.path.join(run_paths["path_logs_train"], "sweep.log")
        utils_misc.set_loggers(log_file_path, logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(["configs/config.gin"], bindings)
        utils_params.save_config(run_paths["path_gin"], gin.config_str())

        # Prepare images and labels
        (
            train_image_paths,
            train_labels,
            val_image_paths,
            val_labels,
            test_image_paths,
            test_labels,
            num_classes,
            label,
        ) = prepare_image_paths_and_labels()

        # Create TF files
        create_tfrecord(train_image_paths, train_labels, "train.tfrecord")
        create_tfrecord(val_image_paths, val_labels, "val.tfrecord")
        create_tfrecord(test_image_paths, test_labels, "test.tfrecord")
        print("TfRecord files created.")

        # setup pipeline
        ds_train, ds_val, ds_test = load()

        # model
        # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
        model = simple_cnn(input_shape=(256, 256, 3), n_classes=2)
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    "name": "mnist-example-sweep",
    "method": "random",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "Trainer.total_steps": {"values": [10]},
        "simple_cnn.base_filters": {
            "distribution": "q_log_uniform",
            "q": 1,
            "min": math.log(8),
            "max": math.log(128),
        },
        "simple_cnn.n_blocks": {
            "distribution": "q_uniform",
            "q": 1,
            "min": 2,
            "max": 6,
        },
        "simple_cnn.dense_units": {
            "distribution": "q_log_uniform",
            "q": 1,
            "min": math.log(16),
            "max": math.log(256),
        },
        "simple_cnn.dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.9},
    },
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=50)

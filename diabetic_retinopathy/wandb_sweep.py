import logging
import wandb
import gin
import math
from input_pipeline.datasets import load
from models.architectures import vgg_like, simple_cnn
from models.transferlearning import  resnet50, densenet201
from train import Trainer
from utils import utils_params, utils_misc
from input_pipeline.createTFRecord import create_tfrecord
from input_pipeline.createTFRecord import prepare_image_paths_and_labels
import os

model_name = 'simple_cnn'
def train_func():
    with wandb.init(project="sweep_diabetic") as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f"{key}={value}")

        # generate folder structures
        run_paths = utils_params.gen_run_folder(",".join(bindings))

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
        if model_name == 'VGG16':
            model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
        elif model_name == 'simple_cnn':
            model = simple_cnn(input_shape=(256, 256, 3), n_classes=2)
        elif model_name == 'resnet50':
            model = resnet50(input_shape=(256, 256, 3), n_classes=2)
        elif model_name == 'densenet201':
            model = densenet201(input_shape=(256, 256, 3), n_classes=2)

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # train the model
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    "name": "idrid-sweep",
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "preprocess.img_height": {"value": 256},
        "preprocess.img_width": {"value": 256},
        "Trainer.total_steps": {"values": [10000]},
        "simple_cnn.base_filters": {
            "values": [8, 16, 32, 64]
        },
        "simple_cnn.n_blocks": {
            "values": [1, 2, 3, 4]
        },
        "simple_cnn.dense_units": {
            "values": [16, 32, 64, 128, 256]
        },
        "simple_cnn.dropout_rate": {"values": [0.1, 0.2, 0.3, 0.4]},
    },
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=50)

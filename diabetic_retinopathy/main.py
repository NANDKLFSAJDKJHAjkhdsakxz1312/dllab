import gin
import logging
from absl import app, flags
from train import Trainer
from train_regression import Trainer_regression
from evaluation.eval import evaluate, evaluate_regression
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like, simple_cnn, simple_cnn_regression
from models.transferlearning import resnet50, densenet121
import os
import wandb

model_name = 'resnet50'
folder = 'resnet50'
FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", True, "Specify whether to train or evaluate a model.")
flags.DEFINE_string("mode", "binary", "Specify the classification mode: binary or multiclass.")

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(folder)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    # setup wandb
    wandb.init(
        project="diabetic_retinopathy",
        name=run_paths["path_model_id"],
        config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG)
    )

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    print("Datasets loaded.")

    if FLAGS.mode == "binary":
        num_classes = 2
        labels = ['0', '1']
    elif FLAGS.mode == "multi":
        num_classes = 5
        labels = ['0', '1', '2', '3', '4']
    else:
        raise ValueError("Unsupported mode. Choose 'binary' or 'multiclass'.")

    # model
    if model_name == 'vgg_Like':
        model = vgg_like(input_shape=(256, 256, 3), num_classes=num_classes)
    elif model_name == 'simple_cnn':
        model = simple_cnn(input_shape=(256, 256, 3), num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(input_shape=(256, 256, 3), num_classes=num_classes)
    elif model_name == 'densenet121':
        model = densenet121(input_shape=(256, 256, 3), num_classes=num_classes)
    elif model_name == 'simple_cnn_regression':
        model = simple_cnn_regression(input_shape=(256, 256, 3))
    print("Model initialized.")


    if FLAGS.train:
        print("Starting training...")
        # set training loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        # train
        if model_name == 'simple_cnn_regression':
            trainer_regression = Trainer_regression(model, ds_train, ds_val, run_paths)
            for _ in trainer_regression.train():
                continue
            trainer_regression.save_model()
        else:
            trainer = Trainer(model, ds_train, ds_val, run_paths)
            for _ in trainer.train():
                continue
            trainer.save_model()

    else:
        print("Starting evaluation...")
        # set evaluation loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        checkpoint_paths = run_paths["path_ckpts_train"]
        if model_name == 'simple_cnn_regression':
            evaluate_regression(model, ds_test, checkpoint_paths)
        else:
            evaluate(model, ds_test, checkpoint_paths, num_classes, labels)


if __name__ == "__main__":
    app.run(main)

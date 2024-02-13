import gin
import logging
import pandas as pd
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate, visualization
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import rnn_model, gru_model, lstm_model
import os
import wandb

model_name = 'gru'
folder = 'gru_model'
FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", False, "Specify whether to train or evaluate a model.")


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(folder)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    # setup wandb
    wandb.init(
        project="human_activity_recognition",
        name=run_paths["path_model_id"],
        config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG)
    )

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()
    print("Datasets loaded.")

    # model
    if model_name == 'rnn':
        model = rnn_model(input_shape=(250, 6), num_classes=12)
    elif model_name == 'gru':
        model = gru_model(input_shape=(250, 6), num_classes=12)
    elif model_name == 'lstm':
        model = lstm_model(input_shape=(250, 6), num_classes=12)
    print("Model initialized.")

    if FLAGS.train:
        print("Starting training...")
        # set training loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
        trainer.save_model()

    else:
        print("Starting evaluation...")
        checkpoint_paths = run_paths["path_ckpts_train"]
        evaluate(model, ds_test, checkpoint_paths)
        visualization(model_name, model, ds_test, checkpoint_paths)

if __name__ == "__main__":
    app.run(main)

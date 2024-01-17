import gin
import logging
import pandas as pd
from absl import app, flags
#from train import Trainer
#from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
#from models.architectures import vgg_like,simple_cnn
import os
import wandb


FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", False, "Specify whether to train or evaluate a model.")


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    log_file_path = os.path.join(run_paths["path_logs_train"], "training.log")
    utils_misc.set_loggers(log_file_path, logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    # # setup wandb
    # wandb.init(
    #     project="human_activity_recognition",
    #     name=run_paths["path_model_id"],
    #     config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG)
    # )

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()
    print("Datasets loaded.")

    # # model
    # model = simple_cnn(input_shape=(250, 6), n_classes=12)
    # print("Model initialized.")
    #
    # if FLAGS.train:
    #     print("Starting training...")
    #     trainer = Trainer(model, ds_train, ds_val, run_paths)
    #     for _ in trainer.train():
    #         continue
    #     trainer.save_model()
    #
    # else:
    #     print("Starting evaluation...")
    #     evaluate(model, ds_test, run_paths, num_classes, label)


if __name__ == "__main__":
    app.run(main)

import gin
import logging
from absl import app, flags
from train import Trainer
from input_pipeline_s2l import datasets
from utils import utils_params, utils_misc
from architectures.models_crnn import crnn_model
from architectures.models_lstm import lstm_model
from input_pipeline_s2l.preprocessing import preprocessor
from eval import evaluate
import wandb

model_name = 'lstm'
folder = 'lstm'
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',False, 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    wandb.init()
    run_paths = utils_params.gen_run_folder(folder)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    _, _ = preprocessor()

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()
    print("Datasets loaded.")
    # model
    if model_name == 'lstm':
        model = lstm_model(input_shape=(250, 6), num_classes=12)
    elif model_name == 'crnn':
        model = crnn_model(input_shape=(250, 6), num_classes=12)
    print("Model initialized.")

    if FLAGS.train:
        print("Starting training...")
        # set training loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    else:
        print("Starting evaluation...")
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        checkpoint_paths = run_paths["path_ckpts_train"]
        evaluate(model, checkpoint_paths, ds_test)


if __name__ == "__main__":
    app.run(main)

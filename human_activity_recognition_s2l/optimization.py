import logging
import wandb
import gin
from train import Trainer
from input_pipeline_s2l.datasets import load
from input_pipeline_s2l.preprocessing import preprocessor
from architectures.models_lstm import lstm_model
from utils import utils_params, utils_misc


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder(run.id)

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        _, _ = preprocessor()

        # setup pipeline
        ds_train, ds_val, ds_test = load()

        # model
        model = lstm_model(input_shape=(250, 6), num_classes=12)

        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'hapt_tuning_3',
    'method': 'grid',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'num_dense_units': {
            'values': [32,64]
        },
        'num_lstm_units': {
            'values': [32, 64]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="hapt_2_tuning")

wandb.agent(sweep_id, function=train_func)

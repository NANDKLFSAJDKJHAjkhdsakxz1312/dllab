import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like,simple_cnn
from input_pipeline.createTFRecord import create_tfrecord
from input_pipeline.createTFRecord import prepare_image_paths_and_labels
import os
import wandb


FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", True, "Specify whether to train or evaluate a model.")


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    log_file_path = os.path.join(run_paths["path_logs_train"], "training.log")
    utils_misc.set_loggers(log_file_path, logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    # setup wandb
    wandb.init(
        project="diabetic_retinopathy",
        name=run_paths["path_model_id"],
        config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG)
    )

    # Prepare images and labels
    (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels,
        test_image_paths,
        test_labels,
        num_classes,
        label
    ) = prepare_image_paths_and_labels()

    # Create TF files
    create_tfrecord(train_image_paths, train_labels, "train.tfrecord")
    create_tfrecord(val_image_paths, val_labels, "val.tfrecord")
    create_tfrecord(test_image_paths, test_labels, "test.tfrecord")
    print("TfRecord files created.")

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()
    print("Datasets loaded.")

    # model
    #model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    model = simple_cnn(input_shape=(256, 256, 3), n_classes=2)
    print("Model initialized.")

    if FLAGS.train:
        print("Starting training...")
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
        trainer.save_model()

    else:
        print("Starting evaluation...")
        checkpoint_paths = run_paths["path_ckpts_train"]
        evaluate(model, ds_test, checkpoint_paths, num_classes, label)


if __name__ == "__main__":
    app.run(main)

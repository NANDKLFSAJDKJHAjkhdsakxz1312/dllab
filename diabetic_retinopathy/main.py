import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from diabetic_retinopathy.input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from diabetic_retinopathy.input_pipeline.createTFRecord import create_tfrecord
from diabetic_retinopathy.input_pipeline.createTFRecord import prepare_image_paths_and_labels


FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", True, "Specify whether to train or evaluate a model.")


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths["path_logs_train"], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    utils_params.save_config(run_paths["path_gin"], gin.config_str())

    # Prepare images and labels
    (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels,
        test_image_paths,
        test_labels,
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
    model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    print("Model initialized.")

    if FLAGS.train:
        print("Starting training...")
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    else:
        print("Starting evaluation...")
        evaluate(model, ds_test, run_paths)


if __name__ == "__main__":
    app.run(main)

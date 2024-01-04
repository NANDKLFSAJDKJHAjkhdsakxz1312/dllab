from diabetic_retinopathy.visualization.gradcam import GradCAM
from absl import app
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import imutils
import cv2
import tensorflow as tf
from diabetic_retinopathy.utils import utils_params
import os


def main(argv):
    image_path = input("Please input image: ")
    print("[INFO] loading model...")
    run_paths = utils_params.gen_run_folder()
    checkpoint_paths = run_paths["path_ckpts_train"]
    saved_model_path = os.path.join(checkpoint_paths, "saved_model_cnn")
    model = tf.keras.models.load_model(saved_model_path)

    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (256, 256))

    # load the input image from disk (in Keras/TensorFlow format) and preprocess it
    image = load_img(image_path, target_size=(256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # use the network to make predictions on the input image and find the class label index with the largest corresponding probability
    preds = model.predict(image)
    i = np.argmax(preds[0])

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    label = "Predicted Class"
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(
        output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)
    cv2.imshow("Output", output)
    cv2.imwrite("output_image.jpg", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)

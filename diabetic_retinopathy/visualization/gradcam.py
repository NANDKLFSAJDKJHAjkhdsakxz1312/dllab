import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img
import cv2
from guidedBacprop import GuidedBackprop,deprocess_image

def grad_cam(model, images,original_image):

    # extract the corresponding layer result from the model
    grad_cam_model = tf.keras.models.Model([model.inputs], [model.get_layer('conv2d_2').output, model.get_layer('dense_1').output])

    # calculate the gradients of the predictions to the feature maps in the last convolutional layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_model(images, training=False)
        tape.watch(conv_outputs)
        idx = np.argmax(predictions[0])
        top_class = predictions[:, idx]
    grads = tape.gradient(top_class, conv_outputs)

    # calculate the CAM
    weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    output = conv_outputs.numpy()[0]
    for i, w in enumerate(weights):
        output[:, :, i] *= w

    cam = np.mean(output, axis=-1)
    cam = cv2.resize(cam, (images.shape[1], images.shape[2]))

    # normalize the CAM
    cam = np.maximum(cam, 0) / (cam.max() + 1e-16)
    cam_image = cam * 255
    cam_image = np.clip(cam_image, 0, 255).astype('uint8')

    # generate the CAM and Grad-CAM images
    cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_RAINBOW)
    grad_cam_image = cv2.addWeighted(original_image, 0.5, cam_image, 0.5, 0)

    return grad_cam_image, cam, idx


saved_model_path = '/Users/zhouzexu/Documents/6_codePractice/04_dl/gradcam/saved_model'
model = tf.keras.models.load_model(saved_model_path)
model.summary()

image = np.array(load_img("IDRiD_022.jpg"))
image_size_orig = np.shape(image)[0:2]
image_size_resize = (image_size_orig[:]/np.max(image_size_orig[:])*256).astype("int32")
image = tf.cast(image, tf.float32) / 255.0
image = tf.image.resize_with_pad(image, 256, 256)
# image_orig = (image.numpy()*255.0).astype("uint8")
image = image.numpy().astype("uint8")
image = np.expand_dims(image, axis=0)

# Grad_cam
image_orig = cv2.cvtColor(cv2.imread("IDRiD_022.jpg"),cv2.COLOR_BGR2RGB)
image_orig = tf.cast(image_orig, tf.float32)
image_orig = tf.image.resize_with_pad(image_orig, 256, 256)
image_orig = image_orig.numpy().astype("uint8")
grad_cam_image, cam, idx = grad_cam(model, image,image_orig)

# Guided backpropagation
guidedBP = GuidedBackprop(model=model,layerName="conv2d_2")
gb = guidedBP.guided_backprop(image)
gb_im = deprocess_image(gb)
gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)

# Guided grad-cam
cam_expand = np.dstack((cam,cam,cam)) # size: [256,256] to [256,256,3]
guided_grad_cam = np.multiply(cam_expand,gb_im).astype("uint8")

# Plot result
fig, axs = plt.subplots(1,4)
fig.set_figheight(3)
fig.set_figwidth(13)
# Crop the picture from [256,256] to [170,256]
image_orig = image_orig[(256-image_size_resize[0])//2:(256+image_size_resize[0])//2,:]
axs[0].imshow(image_orig)
axs[0].axis('off')
axs[0].set_title("Original Image")
grad_cam_image = grad_cam_image[(256-image_size_resize[0])//2:(256+image_size_resize[0])//2,:]
axs[1].imshow(grad_cam_image)
axs[1].axis('off')
axs[1].set_title("Grad-CAM")
gb_im = gb_im[(256-image_size_resize[0])//2:(256+image_size_resize[0])//2,:]
axs[2].imshow(gb_im)
axs[2].axis('off')
axs[2].set_title("Guided Backpropagation")
guided_grad_cam = guided_grad_cam[(256-image_size_resize[0])//2:(256+image_size_resize[0])//2,:]
axs[3].imshow(guided_grad_cam)
axs[3].axis('off')
axs[3].set_title("Guided Grad-CAM")
plt.show()
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# this function will open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_nrows, img_ncols):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def plot_results(
    best_img, base_image_path, style_image_path, img_nrows, img_ncols, CombinedPath
):

    imgx = deprocess_image(best_img.copy(), img_nrows, img_ncols)

    plt.figure(figsize=(50, 50))
    plt.subplot(5, 5, 1)
    plt.title("Base Image", fontsize=20)
    img_base = load_img(base_image_path)
    plt.imshow(img_base)

    plt.subplot(5, 5, 1 + 1)
    plt.title("Style Image", fontsize=20)
    img_style = load_img(style_image_path)
    plt.imshow(img_style)

    plt.subplot(5, 5, 1 + 2)
    plt.title("Final Image", fontsize=20)
    plt.imshow(imgx)

    plt.savefig(CombinedPath + "/FINAL_3_results.png")

    plt.show()

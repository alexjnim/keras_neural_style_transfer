import argparse
import os.path

from keras import backend as K
from keras.applications.vgg19 import VGG19
from resources.image_processing_components import *
from resources.nst_components import *
import tensorflow as tf
from tqdm import tqdm
from scipy.optimize import fmin_l_bfgs_b

tf.compat.v1.disable_eager_execution()


def get_combined_loss(
    args,
    model,
    combined_image,
    img_nrows,
    img_ncols,
):

    # can check all the layers in the model with [layer.name for layer in model.layers]

    # content layer we are interested in
    content_layer = "block5_conv4"
    # content_layer = "block5_conv2"

    # Style layer we are interested in
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
        "block1_conv2",
        "block2_conv2",
        "block3_conv2",
        "block4_conv2",
        # "block5_conv2",
    ]

    # get all the layers from model
    layer_outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # get combined loss as a single scalar
    loss = K.variable(0.0)
    # get content loss from content layer
    content_features = layer_outputs_dict[content_layer]
    content_image_content_features = content_features[0, :, :, :]
    combined_image_content_features = content_features[2, :, :, :]
    loss = loss + args.CONTENT_WEIGHT * get_content_loss(
        content_image_content_features, combined_image_content_features
    )

    # get style loss from all style layers
    for layer_name in style_layers:
        layer_features = layer_outputs_dict[layer_name]
        style_image_style_features = layer_features[1, :, :, :]
        combined_image_style_features = layer_features[2, :, :, :]
        style_loss = get_style_loss(
            style_image_style_features,
            combined_image_style_features,
            img_nrows,
            img_ncols,
        )
        loss = loss + (args.STYLE_WEIGHT / len(style_layers)) * style_loss

    # get variation loss
    loss = loss + args.TOTAL_VARIATION_WEIGHT * total_variation_loss(
        combined_image, img_nrows, img_ncols
    )
    return loss


def run_style_transfer(
    args,
):

    content_image_path = os.path.join("images/content_images/", args.CONTENT_IMAGE_NAME)
    style_image_path = os.path.join("images/style_images/", args.STYLE_IMAGE_NAME)
    combined_folder_path = os.path.join(
        "images",
        "combined_images",
        args.CONTENT_IMAGE_NAME.split(".")[0]
        + "_c_"
        + args.STYLE_IMAGE_NAME.split(".")[0],
    )

    if not os.path.exists(combined_folder_path):
        os.mkdir(combined_folder_path)

    # dimensions of the generated picture.
    width, height = load_img(content_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    content_image = K.variable(
        preprocess_image(content_image_path, img_nrows, img_ncols)
    )
    style_image = K.variable(preprocess_image(style_image_path, img_nrows, img_ncols))

    # this will contain our generated image
    if K.image_data_format() == "channels_first":
        combined_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combined_image = K.placeholder((1, img_nrows, img_ncols, 3))

    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([content_image, style_image, combined_image], axis=0)

    model = VGG19(input_tensor=input_tensor, include_top=False, weights="imagenet")

    # get the combined loss function
    loss = get_combined_loss(
        args,
        model,
        combined_image,
        img_nrows,
        img_ncols,
    )

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combined_image)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    f_outputs = K.function([combined_image], outputs)
    x_opt = preprocess_image(content_image_path, img_nrows, img_ncols)

    # create evaluator to generate loss and gradients
    evaluator = Evaluator(img_nrows, img_ncols, f_outputs)

    # create variables for storing best results here
    best_loss, best_iteration, best_img = float("inf"), 0, None

    for i in tqdm(range(args.ITERATIONS)):
        # run the gradient-ascent process using SciPy’s L-BFGS algorithm,
        # saving the current generated image at each iteration of the algorithm
        x_opt, min_val, info = fmin_l_bfgs_b(
            evaluator.loss,
            x_opt.flatten(),  # must be flattened for L-BFGS
            fprime=evaluator.grads,
            maxfun=20,
            disp=True,
        )

        # saving results every 5 interations
        if i % 5 == 0:
            imgx = deprocess_image(x_opt.copy(), img_nrows, img_ncols)
            plt.imsave(combined_folder_path + "/intermediate_result_%d.png" % i, imgx)

        if min_val < best_loss:
            # Update best loss and best image from total loss.
            print(f"iteration {i} has the best loss")
            best_loss = min_val
            best_iteration = i
            best_img = x_opt.copy()

    best = deprocess_image(best_img.copy(), img_nrows, img_ncols)
    final_image_path = (
        combined_folder_path + "/FINAL_IMAGE_" + str(best_iteration) + ".png"
    )
    plt.imsave(final_image_path, best)

    plot_results(
        best_img,
        content_image_path,
        style_image_path,
        img_nrows,
        img_ncols,
        combined_folder_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--CONTENT_IMAGE_NAME",
        "-c",
        default="doors.jpg",
        action="store",
        type=str,
        help="",
    )
    parser.add_argument(
        "--STYLE_IMAGE_NAME",
        "-s",
        default="monet_iris.jpeg",
        action="store",
        type=str,
        help="",
    )

    parser.add_argument(
        "--ITERATIONS",
        "-i",
        default=200,
        action="store",
        type=int,
        help="",
    )

    parser.add_argument(
        "--CONTENT_WEIGHT",
        "-cw",
        default=1,
        action="store",
        type=int,
        help="",
    )

    parser.add_argument(
        "--STYLE_WEIGHT",
        "-sw",
        default=100,
        action="store",
        type=int,
        help="",
    )

    parser.add_argument(
        "--TOTAL_VARIATION_WEIGHT",
        "-tvw",
        default=20,
        action="store",
        type=int,
        help="",
    )

    args = parser.parse_args()
    run_style_transfer(args)

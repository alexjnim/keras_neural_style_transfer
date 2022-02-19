import numpy as np
from keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class Evaluator:
    # set up a Python class named Evaluator that computes both the loss value and the gradients value at once,
    # returns the loss value when called the first time, and caches the gradients for the next call.
    def __init__(self, img_nrows, img_ncols, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols
        self.f_outputs = f_outputs

    def loss(self, x):
        assert self.loss_value is None

        loss_value, grad_values = eval_loss_and_grads(
            x, self.img_nrows, self.img_ncols, self.f_outputs
        )
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def get_content_loss(base_content, target):
    return K.sum(K.square(target - base_content))


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(input_tensor):
    assert K.ndim(input_tensor) == 3
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram  # /tf.cast(n, tf.float32)


def get_style_loss(style, combination, img_nrows, img_ncols):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C))  # /(4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x, img_nrows, img_ncols):
    a = K.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = K.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x, img_nrows, img_ncols, f_outputs):
    if K.image_data_format() == "channels_first":
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))

    outs = f_outputs([x])
    loss_value = outs[0]

    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype("float64")
    else:
        grad_values = np.array(outs[1:]).flatten().astype("float64")

    return loss_value, grad_values

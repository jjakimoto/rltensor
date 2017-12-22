import numpy as np
from scipy.misc import imresize
import tensorflow as tf


epsilon = 1e-6


def sum_keep_shape(x, axis=-1):
    x_sum = tf.reduce_sum(x, axis=axis, keep_dims=True)
    shape = [1] * len(x.get_shape().as_list())
    shape[axis] = x.get_shape().as_list()[axis]
    x_sum = tf.tile(x_sum, shape)
    return x_sum


def resize_data(data, width, height, c_dim=3, is_color=True):
    """resize data for trainining dcgan
    Args:
        data: list of image data, each of which has a shape,
            (width, height, color_dim) if is_color==True
            (width, height) otherwisei
    """
    if is_color:
        converted_data = np.array([imresize(d, [width, height]) for d in data
                                if (len(d.shape)==3 and d.shape[-1] == c_dim)])
    else:
        # gray scale data
        converted_data = np.array([imresize(d, [width, height]) for d in data
                                if (len(d.shape)==2)])
    return converted_data


def get_shape(input_shape, batch_size=None, maxlen=None):
    """Get shape of batch input for model

    Args:
        input_shape: int or tuple, shape of input
        is_batch: bool, if True, we add another batch
            dimension, None

    Returns:
        tuple: if is_batch is False, shape=(None, *input_shape)
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    elif isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    if maxlen:
        input_shape = (maxlen,) + input_shape
    input_shape = (batch_size,) + input_shape
    return input_shape

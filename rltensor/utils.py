import numpy as np
from scipy.misc import imresize
import tensorflow as tf


epsilon = 1e-6


def sum_keep_shape_array(x, axis=-1):
    sum_x = np.sum(x, axis=axis, keepdims=True)
    shape = [1] * len(x.shape)
    shape[axis] = x.shape[axis]
    sum_x = np.tile(sum_x, shape)
    return sum_x


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


def get_shape(input_shape, is_batch=True, is_sequence=False, maxlen=None):
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
    if is_sequence:
        input_shape = (maxlen,) + input_shape
    if is_batch:
        input_shape = (None,) + input_shape
    return input_shape

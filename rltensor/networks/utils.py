import tensorflow as tf


def get_regularizer(name, scale):
    if name is "l2":
        return tf.contrib.layers.l2_regularizer(scale)
    elif name is "l1":
        return tf.contrib.layers.l1_regularizer(scale)
    elif name is "sum":
        return tf.contrib.layers.sum_regularizer(scale)
    else:
        raise NotImplementedError()

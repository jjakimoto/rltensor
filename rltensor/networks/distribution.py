import tensorflow as tf
import numpy as np

from .ff import FeedForward
from rltensor import utils


class Dirichlet(FeedForward):
    def __init__(self, model_params, action_dim,
                 scope_name=None, max_value=1.0, min_value=0.0,
                 *args, **kwargs):
        if scope_name is None:
            scope_name = "dirichlet"
        model_params.append({"name": "dense",
                             "num_units": action_dim})
        super().__init__(model_params, scope_name)
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, x, training=None):
        alphas, _, log_norm = self.get_params(x, training)
        alphas_sum = utils.sum_keep_shape(alphas, axis=-1)
        return alphas / alphas_sum

    def get_params(self, x, training):
        logits = super().__call__(x, training)
        alphas = tf.log(x=(tf.exp(x=logits) + 1.0)) + 1.0

        shape = (-1,) + self.shape
        alphas = tf.reshape(tensor=alphas, shape=shape)
        alphas_sum = tf.reduce_sum(alphas, axis=-1)
        alphas_sum = tf.maximum(x=alphas_sum, y=utils.epsilon)

        log_norm = tf.reduce_sum(tf.lgamma(alphas), axis=-1)\
            - tf.lgamma(alphas_sum)
        return alphas, alphas_sum, log_norm

    def sample(self, x, training=None):
        alphas, _, _ = self.get_params(x, training)
        alphas_sample = tf.random_gamma(shape=(), alpha=alphas)
        alphas_sample_sum = utils.sum_keep_shape(alphas_sample, axis=-1)
        alphas_sample_sum = tf.maximum(x=alphas_sample_sum, y=utils.epsilon)
        sampled = alphas_sample / alphas_sample_sum
        return self.min_value + (self.max_value - self.min_value) * sampled


class Categorical(FeedForward):
    def __init__(self, model_params, action_dim,
                 scope_name=None, *args, **kwargs):
        if scope_name is None:
            scope_name = "categorical"
        self.action_dim = action_dim
        model_params.append({"name": "dense",
                             "num_units": action_dim})
        super().__init__(model_params, scope_name)

    def __call__(self, x, training=None):
        logits = self.get_params(x, training)
        return logits

    def get_params(self, x, training):
        logits = super().__call__(x, training)
        return logits

    def sample(self, x, training=None):
        sampled = tf.random_uniform(tf.shape(x)[:1], 0,
                                    self.action_dim, tf.int32)
        return sampled

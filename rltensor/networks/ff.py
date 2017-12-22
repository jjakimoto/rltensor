import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d
from tensorflow.contrib.layers import conv2d_transpose, flatten

from .core import BaseNetwork
from .utils import get_regularizer


class FeedForward(BaseNetwork):
    def __init__(self, model_params, scope_name,
                 is_sequence=False, *args, **kwargs):
        self.is_sequence = is_sequence
        super().__init__(model_params, scope_name, *args, **kwargs)

    def __call__(self, x, training=True):
        with tf.variable_scope(self.scope_name, reuse=self.reuse):
            if self.is_sequence:
                dim = x.get_shape().as_list()[2:]
                shape = tf.shape(x)
                batch, length = shape[0], shape[1]
                x = tf.reshape(x, [batch * length] + dim)
            for i, params in enumerate(self.model_params):
                with tf.variable_scope('layer_' + str(i)):
                    if "is_flatten" in params and params["is_flatten"]:
                        x = flatten(x)
                    if "drop_rate" in params:
                        x = tf.layers.dropout(x,
                                              rate=params["drop_rate"],
                                              training=training)
                    # reguralizers
                    options = {}
                    if "w_reg" in params:
                        options["weights_regularizer"] =\
                          get_regularizer(*params["w_reg"])
                    if "b_reg" in params:
                        options["biases_regularizer"] =\
                          get_regularizer(*params["b_reg"])
                    if params["name"] == "dense":
                        x = fully_connected(x,
                                            params["num_units"],
                                            activation_fn=None,
                                            reuse=self.reuse,
                                            scope="dense",
                                            **options)
                    elif params["name"] == "conv2d":
                        x = conv2d(x,
                                   params["num_filters"],
                                   params["kernel_size"],
                                   params["stride"],
                                   params["padding"],
                                   scope="conv2d",
                                   reuse=self.reuse,
                                   activation_fn=None,
                                   **options)
                    elif params["name"] == "deconv2d":
                        x = conv2d_transpose(x,
                                             params["num_filters"],
                                             params["kernel_size"],
                                             params["stride"],
                                             params["padding"],
                                             scope="deconv2d",
                                             reuse=self.reuse,
                                             activation_fn=None,
                                             **options)
                    elif params["name"] == "reshape":
                        x = tf.reshape(x, (-1,) + params["reshape_size"])
                    elif params["name"] == "pooling":
                        del params["name"]
                        x = tf.nn.pool(x, **params)
                    elif params["name"] is None:
                        pass
                    else:
                        raise NotImplementedError("No implementation for 'name'={}".format(params["name"]))
                    if "is_batch" in params and params["is_batch"]:
                        x = tf.layers.batch_normalization(x,
                                                          training=training,
                                                          momentum=0.9,
                                                          reuse=self.reuse,
                                                          name="batch_norm")
                    if "activation" in params:
                        x = params["activation"](x)
            if self.is_sequence:
                dim = x.get_shape().as_list()[1:]
                x = tf.reshape(x, [batch, length] + dim)
            if self.reuse is False:
                self.global_scope_name = tf.get_variable_scope().name
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.global_scope_name)
        self.reuse = True
        return x


class DuelingModel(FeedForward):
    def __init__(self, output_dim, model_params=None, scope_name=None, *args, **kwargs):
        self.output_dim = output_dim
        if model_params is None:
            model_params = mlp_conf["model"]
        if scope_name is None:
            scope_name = "dueling"
        super().__init__(model_params, scope_name)
        self.feature_model = FeedForward(model_params, scope_name="feature_network")
        self.advantage_model = FeedForward([{"name": "dense",
                                             "num_units": 512},
                                           {"name": "dense",
                                            "num_units": output_dim},],
                                          scope_name="advantage_network")
        self.state_model = FeedForward([{"name": "dense","num_units": 512},
                                        {"name": "dense","num_units": 1},],
                                         scope_name="state_network")

    def __call__(self, x, training=True):
        with tf.variable_scope(self.scope_name, reuse=self.reuse):
            x = self.feature_model(x, training)
            advantage = self.advantage_model(x, training)
            state = self.state_model(x, training)
            if self.reuse is False:
                self.global_scope_name = tf.get_variable_scope().name
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.global_scope_name)
            mean_advantage = tf.concat([tf.reduce_mean(advantage, axis=1, keep_dims=True)
                                        for _ in range(self.output_dim)],
                                       axis=1)
            advantage = advantage - mean_advantage
            state = tf.concat([state for _ in range(self.output_dim)], axis=1)
        self.reuse = True
        return state + advantage


class MLPModel(FeedForward):
    def __init__(self, model_params, output_dim=None, activation=None,
                 scope_name=None, *args, **kwargs):
        if scope_name is None:
            scope_name = "mlp"
        if output_dim is not None:
            model_params.append({"name": "dense",
                                 "num_units": output_dim,
                                 "activation": activation})
        super().__init__(model_params, scope_name)

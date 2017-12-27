import tensorflow as tf
import time
import os
from copy import deepcopy

from rltensor.processors import DefaultProcessor
from rltensor.executions import RunnerMixin


class Agent(RunnerMixin):
    """Base Agent for RL

    Parameters
    ----------
    action_spec : dict
        Have to define  'type' and 'shape'
    state_spec : dict, optional
        Have to define 'type' and 'shape'
    processor : object, optional
    t_learn_start : int (default 100)
        The step to start learning
    t_update_freq : int (default 1)
        The frequency to update parameters
    optimizer_spec : dict
        Configuration of optimizer. You have to define 'type' that has
        to be one of 'adam', adad', 'gd', etc. And, you can also set
        parameters feeded as tensorflow optimization parameters.
    lr_cpec : dict
        Configuration of learning rate schedule. You have to define
        'lr_init', 'lr_decay_step', and 'lr_decay'.
    min_r : float, optional
        Lower clip of reward
    max_r : float, optional
        Upper clip of reward
    sess : tensorflow session, optional
    tensorboard_dir : str, optional
        Directory to store summary for tensorboard.

    Notes
    -----
    You have to define either of state_spec or processor.
    """

    def __init__(self, env, action_spec,
                 state_spec=None, processor=None,
                 optimizer_spec=None, lr_spec=None,
                 t_learn_start=100, t_update_freq=4,
                 min_r=None, max_r=None, sess=None,
                 env_name="env", tensorboard_dir="./logs",
                 load_file_path=None,
                 is_debug=False, *args, **kwargs):
        self.env = env
        self.env_name = env_name
        self.action_spec = action_spec
        self.action_shape = self.action_spec["shape"]
        assert state_spec or processor,\
            "Have to difine either of state_spec or processor"
        if processor is None:
            processor = DefaultProcessor(state_spec["shape"])
        self.state_spec = state_spec
        self.processor = processor
        self.state_shape = self.processor.get_input_shape()

        self.t_learn_start = t_learn_start
        self.t_update_freq = t_update_freq

        # Optimizer config
        self.optimizer_spec = optimizer_spec
        self.lr_spec = lr_spec

        # reward is in (min_r, max_r)
        self.min_r = min_r
        self.max_r = max_r

        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.tensorboard_dir = tensorboard_dir
        self.load_file_path = load_file_path
        self.is_debug = is_debug

        self._global_step = tf.Variable(0, trainable=False)
        self._num_episode = tf.Variable(0, trainable=False)
        # Build tensorflow network
        st = time.time()
        print("Building tensorflow graph...")
        with self.sess.as_default():
            self.update_step_op = tf.assign(self._global_step,
                                            self._global_step + 1)
            self.update_episode_op = tf.assign(self._num_episode,
                                               self._num_episode + 1)
            self.training_ph = tf.placeholder(tf.bool, (), name="training_ph")
            self.learning_rate_op = self._get_learning_rate(self.lr_spec)
            self._build_graph()
            self.saver = tf.train.Saver()
            with tf.name_scope("summary"):
                self._build_summaries()
            self.sess.run(tf.global_variables_initializer())
            if self.load_file_path is not None:
                self.load_params(self.load_file_path)
        print("Finished building tensorflow graph, spent time:",
              time.time() - st)

    def _build_graph(self):
        raise NotImplementedError()

    def init_update(self):
        pass

    def load_params(self, file_path):
        """Loads parameters of an estimator from a file.

        Parameters
        ----------
        file_path: str
            The path to load the file.
        """
        self.saver.restore(self.sess, file_path)
        print("Model restored.")

    def save_params(self, file_path=None, overwrite=True):
        """Saves parameters of an estimator as a file.

        Parameters
        ----------
        file_path: str
            The path to where the parameters should be saved.
        overwrite: bool
            If `False` and `file_path` already exists, it raises an error.
        """
        if file_path is None:
            if not os.path.isdir("params"):
                os.mkdir("params")
            file_path = "params/model.ckpt"
        if not overwrite:
            _path = ".".join([file_path, "meta"])
            if os.path.isfile(_path):
                raise NameError("%s already exists." % file_path)
        save_path = self.saver.save(self.sess, file_path)
        print("Model saved in file: %s" % save_path)

    def _get_learning_rate(self, spec):
        schedule = tf.train.exponential_decay(
            spec["lr_init"],
            self._global_step,
            spec["lr_decay_step"],
            spec["lr_decay"],
            staircase=True)
        learning_rate_op = tf.maximum(spec["lr_min"], schedule)
        return learning_rate_op

    def _get_optimizer(self, learning_rate, spec):
        spec = deepcopy(spec)
        name = spec["type"].lower()
        del spec["type"]
        if name == "gd":
            opt = tf.train.GradientDescentOptimizer(learning_rate, **spec)
        elif name == "adad":
            opt = tf.train.AdadeltaOptimizer(learning_rate, **spec)
        elif name == "adam":
            opt = tf.train.AdamOptimizer(learning_rate, **spec)
        elif name == "adag":
            opt = tf.train.AdagradOptimizer(learning_rate, **spec)
        elif name == "rmsp":
            opt = tf.train.RMSPropOptimizer(learning_rate, **spec)
        elif name == "mom":
            opt = tf.train.MomentumOptimizer(learning_rate, **spec)
        else:
            raise NotImplementedError("No implementation for name={}".format(name))
        return opt

    def _build_optimization(self, loss, optimizer, variables=None):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_vars = optimizer.compute_gradients(loss, var_list=variables)
            if "grad_clip" in self.optimizer_spec:
                grad_clip = self.optimizer_spec["grad_clip"]
            else:
                grad_clip = None
            if grad_clip is not None:
                grads_vars = [(tf.clip_by_norm(gv[0], clip_norm=grad_clip), gv[1]) for gv in grads_vars]
            optim = optimizer.apply_gradients(grads_vars)
        return optim

    def observe(self, observation, action, reward,
                terminal, info, training, is_store):
        observation, action, reward, terminal =\
            self.processor.preprocess(observation,
                                      action,
                                      reward,
                                      terminal)
        # clip reward into  (min_r, max_r)
        if self.max_r is not None:
            reward = min(self.max_r, reward)
        if self.min_r is not None:
            reward = max(self.min_r, reward)
        return self._observe(observation, action, reward,
                             terminal, info, training, is_store)

    def _observe(self, observation, action, reward,
                 terminal, info, training, is_store):
        raise NotImplementedError("Need to define _observe at a subclass of Agent")

    def predict(self, state, *args, **kwargs):
        action = self._predict(state, *args, **kwargs)
        return action

    def _predict(self, state, *args, **kwargs):
        raise NotImplementedError("Need to define _predict at a subclass of Agent")

    def update_step(self):
        self.sess.run(self.update_step_op)

    def update_episode(self):
        self.sess.run(self.update_episode_op)

    @property
    def global_step(self):
        return self._global_step.eval(session=self.sess)

    @property
    def num_episode(self):
        return self._num_episode.eval(session=self.sess)

    @property
    def learning_rate(self):
        return self.learning_rate_op.eval(session=self.sess)

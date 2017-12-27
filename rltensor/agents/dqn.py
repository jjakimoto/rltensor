import tensorflow as tf
import random

from .agent import Agent
from rltensor.utils import get_shape
from rltensor.networks import Categorical
from rltensor.memories import SequentialMemory, PrioritizedMemory


class DQN(Agent):

    def __init__(self, env, action_spec,
                 state_spec=None, processor=None,
                 critic_spec=None,
                 optimizer_spec=None, lr_spec=None,
                 critic_cls=Categorical, explore_spec=None,
                 memory_limit=10000, window_length=4,
                 is_prioritized=True, batch_size=32, error_clip=1.0,
                 discount=0.99, t_target_q_update_freq=10000, double_q=True,
                 t_learn_start=100, t_update_freq=1,
                 min_r=None, max_r=None, sess=None,
                 env_name="env", tensorboard_dir="./logs",
                 load_file_path=None,
                 is_debug=False, *args, **kwargs):
        self.critic_spec = critic_spec
        self.critic_cls = critic_cls
        self.explore_spec = explore_spec
        self.memory_limit = memory_limit
        self.window_length = window_length
        self.is_prioritized = is_prioritized
        self.memory = self._get_memory(limit=self.memory_limit,
                                       window_length=self.window_length,
                                       is_prioritized=self.is_prioritized)
        self.batch_size = batch_size
        self.error_clip = error_clip
        self.discount = discount
        self.t_target_q_update_freq = t_target_q_update_freq
        self.double_q = double_q

        super(DQN, self).__init__(
            env=env,
            action_spec=action_spec,
            state_spec=state_spec,
            processor=processor,
            optimizer_spec=optimizer_spec,
            lr_spec=lr_spec,
            t_learn_start=t_learn_start,
            t_update_freq=t_update_freq,
            min_r=min_r,
            max_r=max_r,
            sess=sess,
            env_name="env",
            tensorboard_dir=tensorboard_dir,
            load_file_path=load_file_path,
            is_debug=is_debug,
            *args, **kwargs)

    def _build_graph(self):
        """Build all of the network and optimizations"""
        # state shape has to be (batch, length,) + input_dim
        self.state_ph = tf.placeholder(
            tf.float32,
            get_shape(self.state_shape,
                      is_sequence=True,
                      maxlen=self.window_length),
            name='state_ph')
        _state_ph = self.processor.tensor_process(self.state_ph)
        self.target_state_ph = tf.placeholder(
            tf.float32,
            get_shape(self.state_shape,
                      is_sequence=True,
                      maxlen=self.window_length),
            name='target_state_ph')
        _target_state_ph = self.processor.tensor_process(self.target_state_ph)
        # Build critic network
        self.critic = self.critic_cls(self.critic_spec,
                                      self.action_shape,
                                      scope_name="critic")
        self.q_val = self.critic(_state_ph, self.training_ph)
        # Build target critic network
        self.target_critic = self.critic_cls(self.critic_spec,
                                             self.action_shape,
                                             scope_name="target_critic")
        self.target_q_val = self.target_critic(_target_state_ph,
                                               self.training_ph)
        assert self.q_val.get_shape().as_list()[-1] == self.action_shape
        self.max_action = tf.argmax(self.q_val, axis=-1)

        # Build action graph
        self.action_ph = tf.placeholder(tf.int32, (None), name="action_ph")
        action_one_hot = tf.one_hot(self.action_ph, depth=self.action_shape)
        self.action_q_val = tf.reduce_sum(self.q_val * action_one_hot, axis=-1)
        if self.double_q:
            max_one_hot = tf.one_hot(self.max_action, depth=self.action_shape)
            self.target_max_q_val =\
                tf.reduce_sum(self.target_q_val * max_one_hot, axis=-1)
        else:
            self.target_max_q_val = tf.reduce_max(self.target_q_val, axis=-1)
        # Build objective function
        self.terminal_ph = tf.placeholder(tf.bool, (None,), name="terminal_ph")
        self.reward_ph = tf.placeholder(tf.float32, (None,), name="reward_ph")
        zeros = tf.zeros_like(self.target_max_q_val)
        self.target_value = tf.where(self.terminal_ph,
                                     x=zeros,
                                     y=self.discount * self.target_max_q_val)
        self.error = self.reward_ph + self.target_value - self.action_q_val
        # self.error = self.reward_ph + self.target_ph - self.action_q_val
        _error = tf.abs(self.error)
        clipped_error = tf.where(_error < self.error_clip,
                                 0.5 * tf.square(_error),
                                 _error,
                                 name='clipped_error')
        self.critic_loss = tf.reduce_mean(clipped_error, name='critic_loss')

        # Build optimizer
        self.update_target_q_network_op =\
            self._get_update_target_q_network_op()
        self.optimizer = self._get_optimizer(self.learning_rate_op,
                                             self.optimizer_spec)
        # Build critic optimization
        self.critic_optim = self._build_optimization(self.critic_loss,
                                                     self.optimizer,
                                                     self.critic_variables)
        # Action
        self.policy_action = self.max_action
        self.explore_action = self.critic.sample(_state_ph, self.training_ph)
        self.epsilon_tf = self._get_epsilon()

    def _observe(self, observation, action, reward,
                 terminal, info, training, is_store):
        self.memory.append(observation, action, reward,
                           terminal, info, is_store=is_store)

        step = self.global_step

        if training:
            if (step + 1) % self.t_update_freq == 0:
                is_update = True
            else:
                is_update = False
            experiences = self.memory.sample(self.batch_size)
            result = self.q_learning_minibatch(experiences, is_update)
        else:
            result = None

        if (step + 1) % self.t_target_q_update_freq == 0:
            self.update_target_q_network()

        return result

    def q_learning_minibatch(self, experiences, is_update=True):
        feed_dict = {
            self.state_ph: [experience.state0 for experience in experiences],
            self.target_state_ph: [experience.state1
                                   for experience in experiences],
            self.reward_ph: [experience.reward for experience in experiences],
            self.action_ph: [experience.action for experience in experiences],
            self.terminal_ph: [experience.terminal1
                               for experience in experiences],
            self.training_ph: True,
        }
        """
        target_value = self.sess.run(self.target_value, feed_dict=feed_dict)
        feed_dict[self.target_ph] = target_value
        """
        if is_update:
            if self.is_debug:
                self.previous_target_vars =\
                    self.sess.run(self.target_critic_variables)
                self.previoust_vars = self.sess.run(self.critic_variables)
            self.sess.run(self.critic_optim, feed_dict=feed_dict)
            if self.is_debug:
                self.debug_learning()
        q_t, q_max, loss, error = self.sess.run([self.action_q_val,
                                                 self.target_max_q_val,
                                                 self.critic_loss,
                                                 self.error],
                                                feed_dict=feed_dict)
        return q_t, q_max, loss, error, is_update

    def _predict(self, state, ep=None):
        if ep is None:
            ep = self.epsilon
        if random.random() < ep:
            action_tf = self.explore_action
        else:
            action_tf = self.policy_action
        action = self.sess.run(action_tf,
                               feed_dict={self.state_ph: [state],
                                          self.training_ph: False})[0]
        return action

    def _get_update_target_q_network_op(self):
        update_op = []
        for target_var, var in\
                zip(self.target_critic_variables, self.critic_variables):
            update_op.append(tf.assign(target_var, var))
        return update_op

    def _get_memory(self, limit, window_length, is_prioritized):
        if is_prioritized:
            return PrioritizedMemory(limit=limit,
                                     window_length=window_length)
        else:
            return SequentialMemory(limit=limit,
                                    window_length=window_length)

    def _get_epsilon(self):
        t_ep_end = self.explore_spec["t_ep_end"]
        ep_start = self.explore_spec["ep_start"]
        ep_end = self.explore_spec["ep_end"]
        train_steps = tf.cast(self._global_step - self.t_learn_start,
                              tf.float32)
        train_steps = tf.maximum(train_steps, 0.)
        rest_steps = tf.maximum(0., t_ep_end - train_steps)
        delta_ep = max(0, ep_start - ep_end)
        epsilon = ep_end + delta_ep * rest_steps / t_ep_end
        return epsilon

    def update_target_q_network(self):
        self.sess.run(self.update_target_q_network_op)
        if self.is_debug:
            self.debug_update()

    def init_update(self):
        self.update_target_q_network()

    def get_recent_state(self):
        return self.memory.get_recent_state()

    def reset(self):
        self.memory.reset()

    @property
    def critic_variables(self):
        return self.critic.variables

    @property
    def target_critic_variables(self):
        return self.target_critic.variables

    @property
    def epsilon(self):
        return self.epsilon_tf.eval(session=self.sess)

    def debug_update(self):
        target_critic_vars = self.sess.run(self.target_critic_variables)
        critic_vars = self.sess.run(self.critic_variables)
        # Make sure the copy of ciritic network.
        print(len(critic_vars))
        for x, target_x in zip(critic_vars, target_critic_vars):
            assert (x == target_x).all()

    def debug_learning(self):
        # Make sure target network is not updated through optimization
        after_vars = self.sess.run(self.target_critic_variables)
        for after, before in zip(after_vars, self.previous_target_vars):
            assert (after == before).all()
        self.previous_target_vars = after_vars
        # Q network
        after_vars = self.sess.run(self.critic_variables)
        for after, before in zip(after_vars, self.previoust_vars):
            assert (after != before).any()

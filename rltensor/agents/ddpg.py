import numpy as np
import tensorflow as tf
import random

from .agent import Agent
from rltensor.utils import get_shape
from rltensor.networks import MLPModel, FeedForward, Dirichlet
from rltensor.memories import SequentialMemory, PrioritizedMemory


class DDPG(Agent):
    def __init__(self, action_spec,
                 critic_lower_spec, critic_upper_spec, actor_spec,
                 optimizer_spec, lr_spec, state_spec=None, processor=None,
                 critic_lower_cls=FeedForward, critic_upper_cls=MLPModel,
                 actor_cls=Dirichlet, explore_spec=None,
                 memory_limit=10000, window_length=4,
                 is_prioritized=True, batch_size=32, error_clip=1.0,
                 discount=0.99, t_target_q_update_freq=10000, double_q=False,
                 t_learn_start=100, t_update_freq=1,
                 min_r=None, max_r=None, sess=None,
                 tensorboard_dir="./logs"):
        self.critic_lower_spec = critic_lower_spec
        self.critic_upper_spec = critic_upper_spec
        self.actor_spec = actor_spec
        self.critic_lower_cls = critic_lower_cls
        self.critic_upper_cls = critic_upper_cls
        self.actor_cls = actor_cls
        self.explore_spec = explore_spec
        self.memory_limit = memory_limit
        self.window_length = window_length
        self.is_prioritized = is_prioritized
        self.memory = self._get_memory(self.window_length,
                                       self.memory_limit,
                                       self.is_prioritized)
        self.batch_size = batch_size
        self.error_clip = error_clip
        self.discount = discount
        self.t_target_q_update_freq = t_target_q_update_freq
        self.double_q = double_q

        super(DDPG, self).__init__(
            action_spec, optimizer_spec, lr_spec,
            state_spec, processor,
            t_learn_start, t_update_freq,
            min_r, max_r, sess, tensorboard_dir)

    def _build_graph(self):
        """Build all of the network and optimizations"""
        # state shape has to be (batch, length,) + input_dim
        self.state_ph = tf.placeholder(
            tf.float32,
            get_shape(self.state_dim, maxlen=self.window_length),
            name='state_ph')
        _state_ph = self.processor.tensor_process(self.state_ph)
        self.target_state_ph = tf.placeholder(
            tf.float32,
            get_shape(self.state_dim, maxlen=self.window_length),
            name='target_state_ph')
        _target_state_ph = self.processor.tensor_process(self.target_state_ph)
        # Build critic network
        self.critic_lower = self.critic_lower_cls(self.critic_lower_spec,
                                                  scope_name="critic_lower")
        self.critic_upper = self.critic_upper_cls(self.critic_upper_spec,
                                                  output_dim=1,
                                                  scope_name="critic_upper")
        critic_embed = self.critic_lower(_state_ph, self.training)
        # Build target critic network
        self.target_critic_lower = self.critic_lower_cls(
            self.critic_lower_spec,
            scope_name="target_critic_lower")
        self.target_critic_upper = self.critic_upper_cls(
            self.critic_upper_spec,
            output_dim=1,
            scope_name="target_critic_upper")
        target_critic_embed = self.target_critic_lower(_target_state_ph,
                                                       self.training)
        self.action_ph = tf.placeholder(tf.float32,
                                        get_shape(self.action_shape),
                                        name="action_ph")
        sa = tf.concat((critic_embed, self.action_ph), axis=-1)
        self.sa_val = self.critic_upper(sa, self.training)
        # Build actor network
        self.actor = self.actor_cls(self.action_dim,
                                    self.actor_spec,
                                    scope_name="actor")
        self.actor_action = self.actor(_state_ph, self.training)
        self.target_actor_action = self.actor(_target_state_ph, self.training)

        actor_sa = tf.concat((critic_embed, self.actor_action), axis=-1)
        self.state_val = self.critic_uppder(actor_sa, self.training)
        self.state_val = tf.squeeze(self.state_val, axis=-1)
        target_actor_sa = tf.concat(
            (target_critic_embed, self.target_actor_action),
            axis=-1)
        self.target_state_val = self.target_critic_upper(target_actor_sa,
                                                         self.training)
        self.target_state_val = tf.squeeze(self.target_state_val, axis=-1)
        # Build critic objective function
        self.terminal_ph = tf.placeholder(tf.bool, (None,), name="terminal_ph")
        self.reward_ph = tf.placeholder(tf.float32, (None,), name="reward_ph")
        zeros = tf.zeros_like(tensor=self.target_state_value)
        target_value = tf.where(self.terminal_ph,
                                x=zeros,
                                y=self.discount * self.target_state_value)

        self.error = self.reward_ph + target_value - self.state_value
        clipped_error = tf.where(tf.abs(self.error) < self.error_clip,
                                 0.5 * tf.square(self.error),
                                 tf.abs(self.error), name='clipped_error')
        self.critic_loss = tf.reduce_mean(clipped_error, name='loss_critic')
        self.actor_loss = tf.reduce_mean(-self.state_val, name="actor_critic")

        self.target_update_op = self._get_target_update_op()
        self.optimizer = self._get_optimizer(self.optimizer_name,
                                             self.learning_rate_op,
                                             self.optimizer_conf)
        # Build critic optimization
        self.critic_optim = self._build_optimization(self.critic_loss,
                                                     self.critic_variables)
        self.actor_optim = self._build_optimization(self.actor_loss,
                                                    self.actor_variables)

    def observe(self, observation, action, reward,
                terminal, training, is_store):
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
        self.memory.append(observation, action, reward,
                           terminal, is_store=is_store)

        step = self.global_step.eval(session=self.sess)

        if training:
            if (step + 1) % self.t_update_freq == 0:
                is_update = True
            else:
                is_update = False
            self.memory.add_weights()
            weights = self.memory.get_weights()
            experiences = self.memory.sample(self.batch_size, weights)
            weights = self.memory.get_importance_weights()
            if weights is None:
                weights = np.ones(self.batch_size)
            result = self.q_learning_minibatch(experiences, weights, is_update)
        else:
            result = None

        step = self.global_step
        if (step + 1) % self.t_target_q_update_freq == 0:
            self.update_target_q_network()

        return result

    def q_learning_minibatch(self, experiences, batch_weights, is_update):
        feed_dict = {
            self.state: [experience.state0 for experience in experiences],
            self.target_state: [experience.state1 for
                                experience in experiences],
            self.reward: [experience.reward for experience in experiences],
            self.action: [experience.action for experience in experiences],
            self.terminal: [experience.terminal1 for
                            experience in experiences],
            self.training: True,
        }
        if is_update:
            self.sess.run(self.critic_optim, feed_dict=feed_dict)
        q_t, loss, error = self.sess.run([self.q_val,
                                          self.critic_loss,
                                          self.error],
                                         feed_dict=feed_dict)
        return q_t, loss, error, is_update

    def predict(self, state, ep=None):
        if ep is None:
            ep = self.epsilon.eval(session=self.sess)
        if random.random() < ep:
            action = np.random.randint(0, self.action_dim)
        else:
            action = self.sess.run(self.max_action,
                                   feed_dict={self.state: [state],
                                              self.training: False})[0]
        return action

    def _get_target_update_op(self):
        update_op = []
        for target_var, var in\
                zip(self.target_critic_variables, self.critic_variables):
            update_op.append(tf.assign(target_var, var))
        return update_op

    def _get_memory(self, window_length, limit, is_prioritized):
        if is_prioritized:
            return PrioritizedMemory(window_length, limit)
        else:
            return SequentialMemory(window_length, limit)

    def _get_epsilon(self):
        rest_steps  = tf.maximum(0.,
            self.t_ep_end - tf.maximum(0., tf.cast(self.global_step - self.t_learn_start, tf.float32)))
        delta_ep = max(0, self.ep_start - self.ep_end)
        epsilon = self.ep_end + delta_ep * rest_steps / self.t_ep_end
        return epsilon

    def _build_optimization(self, loss, variables=None):
        grads_vars = self.optimizer.compute_gradients(loss, var_list=variables)
        if "grad_clip" in self.optimizer_spec:
            grad_clip = self.optimizer_spec["grad_clip"]
        else:
            grad_clip = None
        if grad_clip is not None:
            grads_vars = [(tf.clip_by_norm(gv[0], clip_norm=grad_clip), gv[1])
                          for gv in grads_vars]
        optim = self.optimizer.apply_gradients(grads_vars)
        return optim

    def update_target_q_network(self):
        self.sess.run(self.target_update_op)

    def init_update(self):
        self.update_target_q_network()

    def get_recent_state(self):
        return self.memory.get_recent_state()

    @property
    def actor_variables(self):
        return self.actor.variables

    @property
    def critic_variables(self):
        return self.critic_lower.variables\
            + self.critic_upper.variables

    @property
    def target_critic_variables(self):
        return self.target_critic_lower.variables\
            + self.target_critic_upper.variables

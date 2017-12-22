import tensorflow as tf

from .agent import Agent
from rltensor.utils import get_shape
from rltensor.networks import Dirichlet
from rltensor.memories import TSMemory


class EIIE(Agent):

    def __init__(self, action_spec,
                 state_spec=None, processor=None,
                 actor_spec=None,
                 optimizer_spec=None, lr_spec=None,
                 actor_cls=Dirichlet, explore_spec=None,
                 memory_limit=10000, window_length=4, beta=0.1,
                 is_prioritized=True, batch_size=32, error_clip=1.0,
                 t_learn_start=100, t_update_freq=1,
                 min_r=None, max_r=None, sess=None, *args, **kwargs):
        self.actor_spec = actor_spec
        self.actor_cls = actor_cls
        self.explore_spec = explore_spec
        self.memory_limit = memory_limit
        self.window_length = window_length
        self.beta = beta
        self.memory = self._get_memory(self.window_length,
                                       self.memory_limit,
                                       self.beta)
        self.batch_size = batch_size

        super(EIIE, self).__init__(
            action_spec,
            state_spec, processor,
            optimizer_spec, lr_spec,
            t_learn_start, t_update_freq,
            min_r, max_r, sess, *args, **kwargs)

    def _build_graph(self):
        """Build all of the network and optimizations"""
        # state shape has to be (batch, length,) + input_dim
        self.state_ph = tf.placeholder(
            tf.float32,
            get_shape(self.state_shape, maxlen=self.window_length),
            name='state_ph')
        _state_ph = self.processor.tensor_process(self.state_ph)
        # Build actor network
        self.actor = self.actor_cls(self.action_dim,
                                    self.actor_spec,
                                    scope_name="actor")
        self.actor_action = self.actor(_state_ph, self.training)
        # Build critic objective function
        self.terminal_ph = tf.placeholder(tf.bool, (None,), name="terminal_ph")
        self.reward_ph = tf.placeholder(tf.float32,
                                        get_shape(self.action_shape),
                                        name="reward_ph")
        actor_returns = tf.reduce_sum(self.actor_action * self.reward_ph,
                                      axis=-1)
        self.actor_value = tf.reduce_mean(tf.log(actor_returns))
        self.actor_loss = -self.actor_value

        # Build optimizer
        self.optimizer = self._get_optimizer(self.learning_rate_op,
                                             self.optimizer_spec)
        # Build actor optimization
        self.actor_optim = self._build_optimization(self.actor_loss,
                                                    self.optimizer,
                                                    self.actor_variables)
        # Action
        self.policy_action = self.actor_action

    def _observe(self, observation, action, reward,
                 terminal, training, is_store):
        self.memory.append(observation, action, reward,
                           terminal, is_store=is_store)

        step = self.global_step

        if training:
            if (step + 1) % self.t_update_freq == 0:
                is_update = True
            else:
                is_update = False
            experiences = self.memory.sample(self.batch_size)
            result = self.learning_minibatch(experiences, is_update)
        else:
            result = None
        return result

    def learning_minibatch(self, experiences, is_update=True):
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
        if is_update:
            self.sess.run(self.actor_optim, feed_dict=feed_dict)
        actor_loss = self.sess.run(self.actor_loss, feed_dict=feed_dict)
        return actor_loss, is_update

    def predict(self, state, *args, **kwargs):
        action = self.sess.run(self.policy_action,
                               feed_dict={self.state_ph: [state],
                                          self.training_ph: False})[0]
        return action

    def _get_memory(self, window_length, limit, beta):
        return TSMemory(window_length, limit)

    def init_update(self):
        pass

    def get_recent_state(self):
        return self.memory.get_recent_state()

    def reset(self):
        self.memory.reset()

    @property
    def actor_variables(self):
        return self.actor.variables

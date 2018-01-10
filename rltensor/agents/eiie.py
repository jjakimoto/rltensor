import tensorflow as tf

from .agent import Agent
from rltensor.utils import get_shape
from rltensor.networks import EIIEFeedForward
from rltensor.memories import TSMemory
from rltensor.processors import TradeProcessor
from rltensor.executions import TradeRunnerMixin


class EIIE(TradeRunnerMixin, Agent):

    def __init__(self, init_pv=100,
                 env=None, action_spec=None,
                 state_spec=None, processor=None,
                 actor_spec=None,
                 optimizer_spec=None, lr_spec=None,
                 actor_cls=EIIEFeedForward, explore_spec=None,
                 memory_limit=10000, window_length=4, beta=5.0e-5,
                 is_prioritized=True, batch_size=32, error_clip=1.0,
                 t_learn_start=100, t_update_freq=1,
                 min_r=None, max_r=None, sess=None,
                 env_name="env", tensorboard_dir="./logs",
                 load_file_path=None,
                 is_debug=False, *args, **kwargs):
        self.init_pv = init_pv
        self.commission_rate = env.commission_rate
        self.actor_spec = actor_spec
        self.actor_cls = actor_cls
        self.explore_spec = explore_spec
        self.memory_limit = memory_limit
        self.window_length = window_length
        self.beta = beta
        self.memory = self._get_memory(limit=self.memory_limit,
                                       window_length=self.window_length,
                                       beta=self.beta)
        self.batch_size = batch_size
        # Make sure having enough data for sampling
        t_learn_start = max(t_learn_start, batch_size + window_length - 1)
        processor = TradeProcessor(state_spec["shape"])
        super(EIIE, self).__init__(
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
        # Build actor network
        self.actor = self.actor_cls(self.actor_spec,
                                    self.action_shape,
                                    scope_name="actor")
        self.prev_action_ph = tf.placeholder(tf.float32,
                                             get_shape(self.action_shape,
                                                       is_sequence=False),
                                             name='prev_action_ph')
        _prev_action_ph = tf.expand_dims(tf.expand_dims(self.prev_action_ph, axis=1), axis=-1)
        self.actor_action = self.actor(_state_ph, self.training_ph,
                                       additional_x=_prev_action_ph)
        # Build critic objective function
        self.terminal_ph = tf.placeholder(tf.bool, (None,), name="terminal_ph")
        self.reward_ph = tf.placeholder(tf.float32,
                                        get_shape(self.action_shape),
                                        name="reward_ph")
        actor_returns = tf.reduce_sum(self.actor_action * self.reward_ph,
                                      axis=-1)
        # index 0 has to be cash
        trade_amount = tf.reduce_sum(tf.abs(self.actor_action[:, 1:] - self.prev_action_ph[:, 1:]), axis=-1)
        reduction_coef = 1. - self.commission_rate * trade_amount
        actor_returns = (1. + actor_returns) * reduction_coef - 1.
        self.actor_value = tf.reduce_mean(tf.log(actor_returns + 1.))
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
            result = self.learning_minibatch(experiences, is_update)
        else:
            result = None
        return result

    def nonobserve_learning(self):
        experiences = self.memory.sample(self.batch_size)
        result = self.learning_minibatch(experiences, is_update=True)
        return result

    def learning_minibatch(self, experiences, is_update=True):
        feed_dict = {
            self.state_ph: [experience.state for experience in experiences],
            self.reward_ph: [experience.reward for experience in experiences],
            self.terminal_ph: [experience.terminal
                               for experience in experiences],
            self.prev_action_ph: [experience.action
                                  for experience in experiences],
            self.training_ph: True,
        }
        if is_update:
            self.sess.run(self.actor_optim, feed_dict=feed_dict)
            actions = self.sess.run(self.actor_action, feed_dict=feed_dict)
            indices = [experience.index for experience in experiences]
            self._update_pvm(actions, indices)
            # print('intermediate_x')
            # print(self.sess.run(self.actor.prev_activation,
            #                     feed_dict=feed_dict)[0])
        actor_loss = self.sess.run(self.actor_loss, feed_dict=feed_dict)
        return actor_loss, is_update

    def predict(self, state, prev_action, *args, **kwargs):
        action = self.sess.run(self.policy_action,
                               feed_dict={self.state_ph: [state],
                                          self.prev_action_ph: [prev_action],
                                          self.training_ph: False})[0]
        return action

    def _get_memory(self, limit, window_length, beta):
        return TSMemory(limit=limit,
                        window_length=window_length,
                        beta=beta)

    def _update_pvm(self, actions, indices):
        for action, idx in zip(actions, indices):
            self.memory.actions[idx] = action

    def init_update(self):
        pass

    def get_recent_state(self):
        return self.memory.get_recent_state()

    def get_recent_actions(self):
        return self.memory.get_recent_actions()

    def reset(self):
        self.memory.reset()

    @property
    def actor_variables(self):
        return self.actor.variables

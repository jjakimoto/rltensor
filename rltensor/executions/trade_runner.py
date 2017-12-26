from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
from six.moves import xrange
import numpy as np
from .runner import RunnerMixin


class TradeRunnerMixin(RunnerMixin):
    def _build_tags(self, scalar_summary_tags=None,
                    histogram_summary_tags=None):
        if scalar_summary_tags is None:
            scalar_summary_tags = [
                'average.loss',
                'average.returns',
                'drawdown',
                'cumulative_returns',
                'episode.final_value',
                'episode.max_returns',
                'episode.min_returns',
                'episode.avg_returns',
                'episode.maximum_drawdowns',
                'episode.sharp_ratio',
                'training.learning_rate',
                'training.num_step_per_sec',
                'training.time',
                'test.cumulative_returns',
                'test.drawdowns',
                'test.final_value',
                'test.maximum_drawdowns',
                'test.sharp_ratio'
            ]
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['episode.returns', 'test.returns']
            for i in range(env.action_dim):
                histogram_summary_tags.append("episode.action_{}".format(i))
                histogram_summary_tags.append("test.action_{}".format(i))
        self.histogram_summary_tags = histogram_summary_tags

    def fit(self, t_max, num_max_start_steps=0,
            save_file_path=None,
            overwrite=True,
            log_freq=1000,
            avg_length=1000,
            init_pv=100.,
            *args, **kwargs):
        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self._reset(self.agent, self.env)

        # Determine if it has to be randomly initialized
        init_flag = True
        # Start from the middle of training
        # t_max = t_max - step
        _env = self.env

        episode_start_time = time.time()

        self.agent.reset()
        state = self.environment.reset()
        accumulated_pv = init_pv
        peak_pv = init_pv
        self.accumulated_pvs = [accumulated_pv]
        draw_downs = [0.]
        returns = [0.]
        self.episode_timestep = 0
        try:
            for t in tqdm(xrange(t_max)):
                if init_flag:
                    init_flag = False
                    if num_max_start_steps == 0:
                        num_random_start_steps = 0
                    else:
                        num_random_start_steps =\
                            np.random.randint(num_max_start_steps)
                    for _ in xrange(num_random_start_steps):
                        action = _env.action_space.sample()
                        observation, reward, terminal, info =\
                            _env.step(action)
                        if terminal:
                            self._reset(self.agent, _env)
                        self.agent.observe(observation, action,
                                           reward, terminal,
                                           training=False, is_store=False)
                # Update step
                self.agent.update_step()
                step = self.agent.global_step
                # 1. predict
                state = self.agent.get_recent_state()
                action = self.agent.predict(state)
                # 2. act
                observation, reward, terminal, info = _env.step(action)
                # 3. store data and train network
                if step < self.agent.t_learn_start:
                    response = self.agent.observe(observation, action, reward,
                                                  terminal, training=False,
                                                  is_store=True)
                    if terminal:
                        self._reset(self.agent, _env)
                else:
                    response = self.agent.observe(observation, action, reward,
                                                  terminal, training=True,
                                                  is_store=True)

                    loss, is_update = response
                    step = self.agent.global_step
                    # update statistics
                    total_reward.append(reward)
                    total_loss.append(loss)
                    total_q_val.append(np.mean(q))
                    total_q_max_val.append(np.mean(q_max))
                    ep_actions.append(action)
                    ep_errors.append(error)
                    ep_rewards.append(reward)
                    ep_losses.append(loss)
                    ep_q_vals.append(np.mean(q))
                    # Write summary
                    if log_freq is not None and step % log_freq == 0:
                        num_per_sec = log_freq / (time.time() - _st)
                        _st = time.time()
                        epsilon = self.agent.epsilon
                        learning_rate = self.agent.learning_rate
                        avg_r = np.mean(total_reward)
                        avg_loss = np.mean(total_loss)
                        avg_q_val = np.mean(total_q_val)
                        avg_q_max_val = np.mean(total_q_max_val)
                        tag_dict = {'episode.num_of_game': self.num_ep,
                                    'average.reward': avg_r,
                                    'average.loss': avg_loss,
                                    'average.q': avg_q_val,
                                    'average.q_max': avg_q_max_val,
                                    'training.epsilon': epsilon,
                                    'training.learning_rate': learning_rate,
                                    'training.num_step_per_sec': num_per_sec,
                                    'training.time': time.time() - self.st}
                        self._inject_summary(tag_dict, step)
                    if terminal:
                        try:
                            cum_ep_reward = np.sum(ep_rewards)
                            max_ep_reward = np.max(ep_rewards)
                            min_ep_reward = np.min(ep_rewards)
                            avg_ep_reward = np.mean(ep_rewards)
                        except:
                            cum_ep_reward = 0
                            max_ep_reward = 0
                            min_ep_reward = 0
                            avg_ep_reward = 0

                        tag_dict = {'episode.cumulative_reward': cum_ep_reward,
                                    'episode.max_reward': max_ep_reward,
                                    'episode.min_reward': min_ep_reward,
                                    'episode.avg_reward': avg_ep_reward,
                                    'episode.rewards': ep_rewards,
                                    'episode.actions': ep_actions,
                                    'episode.errors': ep_errors}
                        self._inject_summary(tag_dict, self.num_ep)
                        # Reset stored current states
                        self._reset(self.agent, _env)
                        ep_rewards = []
                        ep_losses = []
                        ep_q_vals = []
                        ep_actions = []
                        ep_errors = []
                        self.num_ep += 1
                        init_flag = True
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.agent.save_params(save_file_path, True)

    def _record(self, observation, reward, terminal, info,
                action, response, log_freq):
        q, q_max, loss, error, is_update = response
        step = self.agent.global_step
        # update statistics
        self.total_reward.append(reward)
        self.total_loss.append(loss)
        self.total_q_val.append(np.mean(q))
        self.total_q_max_val.append(np.mean(q_max))
        self.ep_actions.append(action)
        self.ep_errors.append(error)
        self.ep_rewards.append(reward)
        self.ep_losses.append(loss)
        self.ep_q_vals.append(np.mean(q))
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            epsilon = self.agent.epsilon
            learning_rate = self.agent.learning_rate
            avg_r = np.mean(self.total_reward)
            avg_loss = np.mean(self.total_loss)
            avg_q_val = np.mean(self.total_q_val)
            avg_q_max_val = np.mean(self.total_q_max_val)
            tag_dict = {'episode.num_of_game': self.num_ep,
                        'average.reward': avg_r,
                        'average.loss': avg_loss,
                        'average.q': avg_q_val,
                        'average.q_max': avg_q_max_val,
                        'training.epsilon': epsilon,
                        'training.learning_rate': learning_rate,
                        'training.num_step_per_sec': num_per_sec,
                        'training.time': time.time() - self.st}
            self._inject_summary(tag_dict, step)
        if log_freq is not None:
            if terminal:
                try:
                    cum_ep_reward = np.sum(self.ep_rewards)
                    max_ep_reward = np.max(self.ep_rewards)
                    min_ep_reward = np.min(self.ep_rewards)
                    avg_ep_reward = np.mean(self.ep_rewards)
                except:
                    cum_ep_reward = 0
                    max_ep_reward = 0
                    min_ep_reward = 0
                    avg_ep_reward = 0

                tag_dict = {'episode.cumulative_reward': cum_ep_reward,
                            'episode.max_reward': max_ep_reward,
                            'episode.min_reward': min_ep_reward,
                            'episode.avg_reward': avg_ep_reward,
                            'episode.rewards': self.ep_rewards,
                            'episode.actions': self.ep_actions,
                            'episode.errors': self.ep_errors}
                self._inject_summary(tag_dict, self.num_ep)
                # Reset stored current states
                self.ep_rewards = []
                self.ep_losses = []
                self.ep_q_vals = []
                self.ep_actions = []
                self.ep_errors = []
                self.num_ep += 1
                self.init_flag = True

    def _record_play(self, observation, reward, terminal, info):
        # accumulate results
        self.ep_rewards.append(reward)
        if terminal:
            try:
                cum_ep_reward = np.sum(self.ep_rewards)
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            except:
                cum_ep_reward = 0
                max_ep_reward = 0
                min_ep_reward = 0
                avg_ep_reward = 0
            self.cum_ep_rewards.append(cum_ep_reward)
            self.max_ep_rewards.append(max_ep_reward)
            self.min_ep_rewards.append(min_ep_reward)
            self.avg_ep_rewards.append(avg_ep_reward)

            tag_dict = {'episode_test.cumulative_rewards': cum_ep_reward,
                        'episode_test.max_rewards': max_ep_reward,
                        'episode_test.min_rewards': min_ep_reward,
                        'episode_test.avg_rewards': avg_ep_reward}
            self._inject_summary(tag_dict, self.num_ep)

    def _build_recorders(self, avg_length):
        self.total_reward = deque(maxlen=avg_length)
        self.total_loss = deque(maxlen=avg_length)
        self.total_q_val = deque(maxlen=avg_length)
        self.total_q_max_val = deque(maxlen=avg_length)
        self.ep_rewards = []
        self.ep_losses = []
        self.ep_q_vals = []
        self.ep_actions = []
        self.ep_errors = []

        self.sp_list = []
        self.mmd_list = []
        self.final_pv_list = []
        self.accumulated_pvs_list = []
        self.draw_downs_list = []
        # accumulate results
        total_reward = deque(maxlen=avg_length)
        total_loss = deque(maxlen=avg_length)
        ep_rewards = []
        ep_losses = []
        ep_q_vals = []
        ep_actions = []
        ep_errors = []
        step = self.agent.global_step
        _st = self.st

    def _build_recorders_play(self, avg_length=None):
        self.ep_rewards = []
        self.cum_ep_rewards = []
        self.max_ep_rewards = []
        self.min_ep_rewards = []
        self.avg_ep_rewards = []

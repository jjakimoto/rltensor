import time
import numpy as np
from collections import deque
from copy import deepcopy
from tqdm import tqdm
from six.moves import xrange

from .runner import RunnerMixin


def calc_sharp_ratio(returns, bench_mark=0, eps=1e-6):
    var = np.var(returns)
    mean = np.mean(returns)
    return (mean - bench_mark) / (np.sqrt(var) + eps)


class TradeRunnerMixin(RunnerMixin):
    def _build_tags(self, scalar_summary_tags=None,
                    histogram_summary_tags=None):
        if scalar_summary_tags is None:
            scalar_summary_tags = [
                'training.average_loss',
                'training.average_returns',
                'training.learning_rate',
                'training.num_step_per_sec',
                'training.time',
                'test.average_loss',
                'test.average_returns',
                'test.learning_rate',
                'test.num_step_per_sec',
                'test.time',
                'test.max_returns',
                'test.min_returns',
                'test.average_returns',
                'test.cumulative_returns',
                'test.drawdown',
                'test.final_value',
                'test.maximum_drawdown',
                'test.sharp_ratio',
            ]
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['test.returns']
            for i in range(self.env.action_dim):
                histogram_summary_tags.append("episode.action_{}".format(i))
                histogram_summary_tags.append("test.action_{}".format(i))
        self.histogram_summary_tags = histogram_summary_tags

    def fit(self, start=None, end=None, num_epochs=100,
            save_file_path=None,
            overwrite=True,
            log_freq=1000,
            avg_length=1000):
        # Save Model
        self.save_params(save_file_path, overwrite)
        # Record Viodeo
        _env = self.env
        _env.set_trange(start, end)
        self._reset(_env, is_reset_memory=True)
        # initialize target netwoork
        self.init_update()
        # accumulate results
        self._build_recorders(avg_length)
        self.st = time.time()
        self.record_st = self.st
        try:
            terminal = False
            step = 0
            cum_time = 0
            while not terminal:
                step += 1
                action = self.actor.sample(1)[0]
                _st = time.time()
                observation, reward, terminal, info =\
                    _env.step(action, is_training=True)
                cum_time += time.time() - _st
                self.observe(observation, action,
                             reward, terminal, info,
                             training=False, is_store=True)
            self.fit_recent_observations =\
                deepcopy(self.memory.recent_observations)
            self.fit_recent_terminals = deepcopy(self.memory.recent_terminals)
            print("Finished storing data.")
            # Start training
            for epoch in tqdm(xrange(num_epochs)):
                # Update step
                self.update_step()
                # Update parameters
                response = self.nonobserve_learning()
                self._record(response, log_freq)
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.save_params(save_file_path, True)

    def play(self, start=None, end=None, num_epochs=1,
             save_file_path=None,
             overwrite=True,
             log_freq=1000,
             avg_length=1000):
        # Save Model
        self.save_params(save_file_path, overwrite)
        # accumulate results
        _env = self.env
        _env.set_trange(start, end)
        observation = self._reset(_env)
        self.memory.set_recent_data(self.fit_recent_observations,
                                    self.fit_recent_terminals)
        self._build_recorders_play(avg_length)
        # Start from the middle of training
        self.st = time.time()
        self.record_st = self.st
        terminal = False
        try:
            while not terminal:
                # 1. predict
                state = self.get_recent_state()
                recent_actions = self.get_recent_actions()
                action = self.predict(state, recent_actions)
                # 2. act
                observation, reward, terminal, info =\
                    _env.step(action, is_training=False)
                self._update_status(observation, reward, terminal, info)
                # 3. store data and train network
                response = self.observe(observation, action, reward,
                                        terminal, info, training=False,
                                        is_store=True)
                for epoch in xrange(num_epochs):
                    # Update step
                    self.update_step()
                    # Update parameters
                    response = self.nonobserve_learning()
                self._record_play(observation, reward, terminal, info,
                                  action, response, log_freq)
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.save_params(save_file_path, True)

    def _update_status(self, observation, reward, terminal, info):
        # Calculate portfolio value
        self.cum_pv *= (1. + reward)
        # Calc drawdown
        if self.cum_pv > self.peak_pv:
            self.peak_pv = self.cum_pv
        self.drawdown = (self.peak_pv - self.cum_pv) / self.peak_pv

    def _reset(self, env, is_reset_memory=False):
        self.drawdown = 0
        self.cum_pv = self.init_pv
        self.peak_pv = self.init_pv
        return super()._reset(env, is_reset_memory=is_reset_memory)

    def _record(self, response, log_freq):
        loss, is_update = response
        step = self.global_step
        # Update statistics
        self.total_losses.append(loss)
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            learning_rate = self.learning_rate
            avg_loss = np.mean(self.total_losses)
            tag_dict = {'training.average_loss': avg_loss,
                        'training.learning_rate': learning_rate,
                        'training.num_step_per_sec': num_per_sec,
                        'training.time': time.time() - self.st}
            self._inject_summary(tag_dict, step)

    def _record_play(self, observation, reward, terminal, info,
                     action, response, log_freq):
        # Update statistics
        self.cum_pv *= (1. + reward)
        # Calc drawdown
        if self.cum_pv > self.peak_pv:
            self.peak_pv = self.cum_pv
        self.drawdown = (self.peak_pv - self.cum_pv) / self.peak_pv

        loss, is_update = response
        step = self.global_step
        # Update statistics
        self.test_returns.append(reward)
        self.test_losses.append(loss)
        self.test_drawdowns.append(self.drawdown)
        self.test_cumulative_returns.append(self.cum_pv)
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            learning_rate = self.learning_rate
            avg_r = np.mean(self.test_returns)
            avg_loss = np.mean(self.test_losses)
            tag_dict = {'test.average_returns': avg_r,
                        'test.average_loss': avg_loss,
                        'test.drawdown': self.drawdown,
                        'test.cumulative_returns': self.cum_pv,
                        'test.learning_rate': learning_rate,
                        'test.num_step_per_sec': num_per_sec,
                        'test.time': time.time() - self.st}
            self._inject_summary(tag_dict, step)
        if log_freq is not None:
            if terminal:
                try:
                    max_returns = np.max(self.test_returns)
                    min_returns = np.min(self.test_returns)
                    avg_returns = np.mean(self.test_returns)
                    max_drawdown = np.max(self.test_drawdowns)
                    sharp_ratio = calc_sharp_ratio(self.test_returns)
                except:
                    max_returns = 0.
                    min_returns = 0.
                    avg_returns = 0.
                    max_drawdown = 0.
                    sharp_ratio = 0.

                tag_dict = {'test.maximum_drawdown': max_drawdown,
                            'test.max_returns': max_returns,
                            'test.min_returns': min_returns,
                            'test.average_returns': avg_returns,
                            'test.returns': self.test_returns,
                            'test.final_value': self.cum_pv,
                            'test.sharp_ratio': sharp_ratio}
                self._inject_summary(tag_dict, self.num_episode)
                self.results = dict(
                    sharp_ratio=sharp_ratio,
                    cumulative_returns=self.test_cumulative_returns,
                    drawdowns=self.test_drawdowns,
                    maximum_drawdows=max_drawdown)

    def _build_recorders(self, avg_length):
        self.total_losses = deque(maxlen=avg_length)

    def _legacy_build_recorders(self, avg_length):
        self.total_returns = deque(maxlen=avg_length)
        self.total_losses = deque(maxlen=avg_length)
        self.ep_returns = []
        self.ep_losses = []
        self.ep_actions = []
        self.ep_cum_pvs = []
        self.ep_drawdowns = []

    def _build_recorders_play(self, avg_length):
        self.test_returns = []
        self.test_losses = []
        self.test_drawdowns = []
        self.test_cumulative_returns = []

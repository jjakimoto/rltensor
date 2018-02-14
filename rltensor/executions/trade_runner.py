import time
import numpy as np
import pandas as pd
from collections import deque
from copy import deepcopy
from tqdm import tqdm
from six.moves import xrange
import matplotlib.pyplot as plt

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
                'test.average_cost',
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
                'test.average_cost',
                'test.cumulative_cost',
            ]
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['test.returns']
            for i in range(self.action_shape):
                histogram_summary_tags.append('test.action_%d' % i)
        self.histogram_summary_tags = histogram_summary_tags

    def fit(self, start=None, end=None, num_epochs=100,
            save_file_path=None,
            overwrite=True,
            log_freq=1000,
            test_freq=100000,
            avg_length=1000,
            test_start=None):
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
        pbar = tqdm()
        try:
            terminal = False
            while not terminal:
                pbar.update(1)
                action = self.actor.sample(1)[0]
                observation, reward, terminal, info =\
                    _env.step(action, is_training=True)
                self.observe(observation, action,
                             reward, terminal, info,
                             training=False, is_store=True)
            self.fit_recent_observations =\
                deepcopy(self.memory.recent_observations)
            self.fit_recent_terminals = deepcopy(self.memory.recent_terminals)
            self.fit_recent_actions = deepcopy(self.memory.recent_actions)
            pbar.close()
            print("Finished storing data.")
            # Start training
            for epoch in tqdm(xrange(num_epochs)):
                # Test
                """
                step = self.global_step
                if test_freq is not None and step % test_freq == 0:
                    self.test_play(start=test_start, end=end, num_epochs=1,
                                   log_freq=None,
                                   avg_length=1000)
                    plt.plot(self.test_cumulative_returns)
                    plt.savefig('test_cumulative_retunrs_%d.jpg' % step)
                    plt.close()
                """
                # Update parameters
                response = self.nonobserve_learning()
                # Update step
                self.update_step()
                self._record(response, log_freq=log_freq)
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.save_params(save_file_path, True)

    def test_play(self, start=None, end=None, num_epochs=1,
                  log_freq=1000,
                  avg_length=1000):
        # accumulate results
        _env = self.env
        _env.set_trange(start, end)
        observation = self._reset(_env, is_reset_memory=False)
        self._build_recorders_play(avg_length)
        # Start from the middle of training
        self.st = time.time()
        self.record_st = self.st
        terminal = False
        pbar = tqdm()
        count = 0
        try:
            while not terminal:
                pbar.update(1)
                # 1. predict
                state = self.get_recent_state()
                recent_actions = self.get_recent_actions()
                action = self.predict(state, recent_actions, is_record=False)
                # 2. act
                observation, reward, terminal, info =\
                    _env.step(action, is_training=False)
                # 3. store data and train network
                response = self.observe(observation, action, reward,
                                        terminal, info, training=False,
                                        is_store=True)
                count += 1
                if count < self.memory.window_length:
                    continue
                self._update_status(observation, reward, terminal, info)
                for epoch in xrange(num_epochs):
                    # Update parameters
                    response = self.nonobserve_learning(use_newest=False)
                # Update step
                self.update_step()
                self._record_play(observation, reward, terminal, info,
                                  action, response, log_freq)
        except KeyboardInterrupt:
            pass
        pbar.close()

    def play(self, start=None, end=None, num_epochs=1,
             save_file_path=None,
             overwrite=True,
             log_freq=1,
             avg_length=1000):
        # Save Model
        self.save_params(save_file_path, overwrite)
        # accumulate results
        _env = self.env
        _env.set_trange(start, end)
        observation = self._reset(_env, is_reset_memory=False)
        self._build_recorders_play(avg_length)
        # Start from the middle of training
        self.st = time.time()
        self.record_st = self.st
        terminal = False
        pbar = tqdm()
        try:
            while not terminal:
                pbar.update(1)
                # 1. predict
                state = self.get_recent_state()
                recent_actions = self.get_recent_actions()
                action = self.predict(state, recent_actions, is_record=True)
                # 2. act
                observation, reward, terminal, info =\
                    _env.step(action, is_training=False)
                self._update_status(observation, reward, terminal, info)
                # 3. store data and train network
                response = self.observe(observation, action, reward,
                                        terminal, info, training=False,
                                        is_store=True)
                for epoch in xrange(num_epochs):
                    # Update parameters
                    # _response = self.nonobserve_learning(use_newest=False)
                    response = self.nonobserve_learning(use_newest=True)
                # Update step
                self.update_step()
                self._record_play(observation, reward, terminal, info,
                                  action, response, log_freq)
        except KeyboardInterrupt:
            pass
        pbar.close()
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
        loss, is_update = response
        step = self.global_step
        # Update statistics
        self.test_avg_returns.append(reward)
        self.test_avg_losses.append(loss)
        self.test_avg_costs.append(info['cost'])
        self.test_returns.append(reward)
        self.test_losses.append(loss)
        self.test_costs.append(info['cost'])
        self.test_time_stamp.append(info['time'])
        self.test_dist_actions.append(action)
        self.test_drawdowns.append(self.drawdown)
        self.test_cumulative_returns.append(self.cum_pv)
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            learning_rate = self.learning_rate
            avg_r = np.mean(self.test_avg_returns)
            avg_loss = np.mean(self.test_avg_losses)
            avg_cost = np.mean(self.test_avg_costs)
            cum_cost = np.sum(self.test_costs)
            tag_dict = {'test.average_returns': avg_r,
                        'test.average_loss': avg_loss,
                        'test.average_cost': avg_cost,
                        'test.drawdown': self.drawdown,
                        'test.cumulative_returns': self.cum_pv,
                        'test.cumulative_cost': cum_cost,
                        'test.learning_rate': learning_rate,
                        'test.num_step_per_sec': num_per_sec,
                        'test.time': time.time() - self.st}
            # Record statistics with certain frequency
            if step % (log_freq * 100) == 0:
                sharp_ratio = calc_sharp_ratio(self.test_avg_returns)
                tag_dict['test.sharp_ratio'] = sharp_ratio
                for i in range(self.action_shape):
                    name = 'test.action_%d' % i
                    tag_dict[name] = np.array(self.test_dist_actions)[:, i]
                self.test_dist_actions = []
                tag_dict['test.returns'] = self.test_avg_returns

            self._inject_summary(tag_dict, step)
        if log_freq is not None:
            if terminal:
                try:
                    max_returns = np.max(self.test_returns)
                    min_returns = np.min(self.test_returns)
                    avg_returns = np.mean(self.test_returns)
                    max_drawdown = np.max(self.test_drawdowns)
                    final_sharp_ratio = calc_sharp_ratio(self.test_returns)
                except:
                    max_returns = 0.
                    min_returns = 0.
                    avg_returns = 0.
                    max_drawdown = 0.
                    final_sharp_ratio = 0.
                tag_dict = {'test.maximum_drawdown': max_drawdown,
                            'test.max_returns': max_returns,
                            'test.min_returns': min_returns,
                            'test.average_returns': avg_returns,
                            'test.final_value': self.cum_pv}
                self._inject_summary(tag_dict, self.num_episode)
                time_idx = pd.DatetimeIndex(self.test_time_stamp)
                self.results = dict(
                    sharp_ratio=final_sharp_ratio,
                    cumulative_returns=pd.DataFrame(self.test_cumulative_returns, index=time_idx),
                    drawdowns=pd.DataFrame(self.test_drawdowns, index=time_idx),
                    returns=pd.DataFrame(self.test_returns, index=time_idx),
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
        self.test_avg_returns = deque(maxlen=avg_length)
        self.test_avg_losses = deque(maxlen=avg_length)
        self.test_avg_costs = deque(maxlen=avg_length)
        self.test_returns = []
        self.test_losses = []
        self.test_drawdowns = []
        self.test_cumulative_returns = []
        self.test_costs = []
        self.test_actions = []
        self.test_dist_actions = []
        self.test_time_stamp = []
        self.test_price_returns = []

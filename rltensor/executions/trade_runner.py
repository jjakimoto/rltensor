import time
import numpy as np
from collections import deque

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
                'training.num_of_game',
                'training.average_loss',
                'training.average_returns',
                'training.drawdown',
                'training.cumulative_returns',
                'training.learning_rate',
                'training.num_step_per_sec',
                'training.time',
                'episode.final_value',
                'episode.max_returns',
                'episode.min_returns',
                'episode.avg_returns',
                'episode.maximum_drawdowns',
                'episode.sharp_ratio',
                'test.cumulative_returns',
                'test.drawdowns',
                'test.final_value',
                'test.maximum_drawdowns',
                'test.sharp_ratio'
            ]
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['episode.returns', 'test.returns']
            for i in range(self.env.action_dim):
                histogram_summary_tags.append("episode.action_{}".format(i))
                histogram_summary_tags.append("test.action_{}".format(i))
        self.histogram_summary_tags = histogram_summary_tags

    def _update_status(self, observation, reward, terminal, info):
        # Calculate portfolio value
        self.cum_pv *= (1. + reward)
        # Calc drawdown
        if self.cum_pv > self.peak_pv:
            self.peak_pv = self.cum_pv
        self.drawdown = (self.peak_pv - self.cum_pv) / self.peak_pv

    def _reset(self, env):
        self.drawdown = 0
        self.cum_pv = self.init_pv
        self.peak_pv = self.init_pv
        return super()._reset(env)

    def _record(self, observation, reward, terminal, info,
                action, response, log_freq):
        loss, is_update = response
        step = self.global_step
        # Update statistics
        self.total_returns.append(reward)
        self.total_losses.append(loss)
        self.ep_returns.append(reward)
        self.ep_losses.append(loss)
        self.ep_actions.append(action)
        self.ep_cum_pvs.append(self.cum_pv)
        self.ep_drawdowns.append(self.drawdown)
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            learning_rate = self.learning_rate
            avg_r = np.mean(self.total_returns)
            avg_loss = np.mean(self.total_losses)
            tag_dict = {'training.num_of_game': self.num_episode,
                        'training.average_returns': avg_r,
                        'training.average_loss': avg_loss,
                        'training.drawdown': self.drawdown,
                        'training.cumulative_returns': self.cum_pv,
                        'training.learning_rate': learning_rate,
                        'training.num_step_per_sec': num_per_sec,
                        'training.time': time.time() - self.st}
            self._inject_summary(tag_dict, step)
        if log_freq is not None:
            if terminal:
                try:
                    max_ep_returns = np.max(self.ep_returns)
                    min_ep_returns = np.min(self.ep_returns)
                    avg_ep_returns = np.mean(self.ep_returns)
                    max_drawdowns = np.max(self.ep_drawdowns)
                    sharp_ratio = calc_sharp_ratio(self.ep_returns)
                except:
                    max_ep_returns = 0
                    min_ep_returns = 0
                    avg_ep_returns = 0
                    max_drawdowns = 0
                    sharp_ratio = 0

                tag_dict = {'episode.maximum_drawdowns': max_drawdowns,
                            'episode.max_returns': max_ep_returns,
                            'episode.min_returns': min_ep_returns,
                            'episode.avg_returns': avg_ep_returns,
                            'episode.returns': self.ep_returns,
                            # 'episode.actions': self.ep_actions,
                            'episode.final_value': self.cum_pv,
                            'episode.sharp_ratio': sharp_ratio}
                self._inject_summary(tag_dict, self.num_episode)
                # Reset stored current states
                self.ep_returns = []
                self.ep_losses = []
                self.ep_drawdowns = []
                self.ep_actions = []
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
        self.total_returns = deque(maxlen=avg_length)
        self.total_losses = deque(maxlen=avg_length)
        self.ep_returns = []
        self.ep_losses = []
        self.ep_actions = []
        self.ep_cum_pvs = []
        self.ep_drawdowns = []

    def _build_recorders_play(self, avg_length=None):
        self.ep_rewards = []
        self.cum_ep_rewards = []
        self.max_ep_rewards = []
        self.min_ep_rewards = []
        self.avg_ep_rewards = []

from copy import deepcopy
from collections import defaultdict
import pandas as pd
import numpy as np

from rltensor.environments.core import Env
from .utils import seconds2date, calculate_pv_after_commission


class TradeEnv(Env):
    def __init__(self, data, start=None, end=None,
                 add_cash=True,
                 price_keys=['open', 'high', 'low', 'weightedAverage'],
                 volume_keys=['volume', 'quoteVolume'],
                 commission_rate=2.5e-3,
                 init_frame=300):
        self.price_keys = price_keys
        self.volume_keys = volume_keys
        self.commission_rate = commission_rate
        data = deepcopy(data)
        self.symbols = list(data.keys())
        # Build imputed data with columns key
        dfs = defaultdict(lambda: [])
        for symbol, val in data.items():
            dates = val["date"].values[init_frame:]
            dates = pd.DatetimeIndex([seconds2date(d) for d in dates])
            df = pd.DataFrame(val.values[init_frame:],
                              index=dates, columns=val.columns)
            df = df.loc[~df.index.duplicated(keep='first')]
            for col in val.columns:
                if col != 'date':
                    dfs[col].append(df[col])
        for col in dfs.keys():
            df = pd.concat(dfs[col], axis=1, keys=self.symbols)
            df.interpolate(method='linear',
                           limit_direction='both',
                           inplace=True)
            dfs[col] = df
        self.dfs_time_index = df.index
        self.dfs = dfs
        self.set_trange(start, end)
        self.num_stocks = len(self.symbols)
        self._reset()

    def set_trange(self, start=None, end=None):
        if start is None:
            self.start = self.dfs_time_index[0]
        else:
            start = pd.Timestamp(start)
            self.start = max(start, self.dfs_time_index[0])
        if end is None:
            self.end = self.dfs_time_index[-1]
        else:
            end = pd.Timestamp(end)
            self.end = min(end, self.dfs_time_index[-1])

        print('start:', self.start)
        print('end:', self.end)
        # Store imputed data with symbol keys
        price_data = {}
        price_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.price_keys:
                df = self.dfs[col][[symbol]]
                df = df.loc[(df.index >= self.start) & (df.index <= self.end)]
                val.append(df.values)
            self.time_index = df.index
            val = np.concatenate(val, axis=1)
            price_data_val.append(np.expand_dims(val, 1))
            price_data[symbol] = pd.DataFrame(val, columns=self.price_keys,
                                              index=self.time_index)

        # Store imputed data with symbol keys
        volume_data = {}
        volume_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.volume_keys:
                df = self.dfs[col][[symbol]]
                df = df.loc[(df.index >= self.start) & (df.index <= self.end)]
                val.append(df.values)
            self.time_index = df.index
            val = np.concatenate(val, axis=1)
            volume_data_val.append(np.expand_dims(val, 1))
            volume_data[symbol] = pd.DataFrame(val, columns=self.volume_keys,
                                               index=self.time_index)

        self.price_data = price_data
        self.price_data_val = np.concatenate(price_data_val, axis=1)
        self.volume_data = volume_data
        self.volume_data_val = np.concatenate(volume_data_val, axis=1)

    def _reset(self):
        self.current_time = self.start
        self.current_step = 0
        observation = self._get_bar()
        self.prev_action = np.array([1.] + list(np.zeros(self.action_dim - 1)))
        return observation

    def _step(self, action, is_training=True, *args, **kwargs):
        current_bars = self._get_bar()
        prev_bars = self._get_prev_bar()
        returns = current_bars['price'][:, 0] / prev_bars['price'][:, 0] - 1.
        # observation = self._get_observation(current_bars)
        observation = current_bars
        terminal = self._get_terminal()
        trade_amount = np.sum(np.abs(action[1:] - self.prev_action[1:]))
        reward = np.sum(returns * action[1:])
        # We do not calculate actual mu for speeding up
        if not is_training:
            mu = calculate_pv_after_commission(action,
                                               self.prev_action,
                                               self.commission_rate)
            reward = mu * (reward + 1.) - 1.
            cost = 1 - mu
        else:
            cost = 0
        self.prev_action = deepcopy(action)
        info = {
            'reward': reward,
            'returns': returns,
            'cost': cost,
            'trade_amount': trade_amount,
            'time': self._get_time(),
        }
        if not terminal:
            # Update bars
            self._update_time()
        return observation, reward, terminal, info

    def _update_time(self):
        self.current_step += 1
        self.current_time = self.time_index[self.current_step]

    def _get_terminal(self):
        return self.current_time >= self.end

    def _get_time(self):
        return self.current_time

    def _get_bar(self):
        bar = {}
        bar['price'] = self.price_data_val[self.current_step]
        bar['volume'] = self.volume_data_val[self.current_step]
        return bar

    def _get_prev_bar(self):
        bar = {}
        prev_idx = max(self.current_step - 1, 0)
        bar['price'] = self.price_data_val[prev_idx]
        bar['volume'] = self.volume_data_val[prev_idx]
        return bar

    @property
    def action_dim(self):
        return self.num_stocks + 1

    @property
    def feature_dim(self):
        return self.price_data_val.shape[-1] + self.volume_data_val.shape[-1]

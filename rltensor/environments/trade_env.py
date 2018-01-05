from copy import deepcopy
from collections import defaultdict
import pandas as pd
import numpy as np

from rltensor.environments.core import Env
from .utils import seconds2date, calculate_pv_after_commission


class TradeEnv(Env):
    def __init__(self, data, start=None, end=None,
                 add_cash=True, keys=["close", "high", "low"],
                 commission_rate=2.5e-3):
        self.keys = keys
        self.commission_rate = commission_rate
        data = deepcopy(data)
        self.symbols = list(data.keys())
        # Build imputed data with columns key
        dfs = defaultdict(lambda: [])
        for symbol, val in data.items():
            dates = val["date"].values
            dates = pd.DatetimeIndex([seconds2date(d) for d in dates])
            df = pd.DataFrame(val.values, index=dates, columns=val.columns)
            df = df.loc[~df.index.duplicated(keep='first')]
            for col in val.columns:
                if col != 'date':
                    dfs[col].append(df[col])
        for col in dfs.keys():
            df = pd.concat(dfs[col], axis=1, keys=self.symbols)
            df.columns = self.symbols
            if add_cash:
                val = np.ones((df.shape[0], 1))
                cash_df = pd.DataFrame(val, index=df.index, columns=['Cash'])
                df = pd.concat([df, cash_df], axis=1)
            df.interpolate(method='linear',
                           limit_direction='both',
                           inplace=True)
            dfs[col] = df
            self.time_index = df.index
        # Store imputed data with symbol keys
        _data = {}
        data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.keys:
                val.append(dfs[col][[symbol]])
            val = np.concatenate(val, axis=1)
            data_val.append(np.expand_dims(val, 1))
            _data[symbol] = pd.DataFrame(val, columns=self.keys,
                                         index=self.time_index)
        self.set_trange(start, end)
        self.dfs = dfs
        self.data = _data
        self.data_val = np.concatenate(data_val, axis=1)
        self.num_stocks = len(self.symbols)
        self._reset()

    def set_trange(self, start=None, end=None):
        if start is None:
            self.start = self.time_index[0]
        else:
            start = pd.Timestamp(start)
            self.start = max(start, self.time_index[0])
        if end is None:
            self.end = self.time_index[-1]
        else:
            end = pd.Timestamp(end)
            self.end = min(end, self.time_index[-1])

    def _reset(self):
        self.current_time = self.start
        self.current_step = list(self.time_index).index(self.start)
        self.prev_bars = self._get_bar()
        # observation = self._get_observation(self.prev_bars)
        observation = self.prev_bars
        self.prev_action = np.array([1.] + list(np.zeros(self.action_dim - 1)))
        return observation

    def _step(self, action, is_training=True, *args, **kwargs):
        current_bars = self._get_bar()
        returns = current_bars[:, 0] / self.prev_bars[:, 0] - 1.
        # Update bars
        self.prev_bars = deepcopy(current_bars)
        self._update_time()
        # observation = self._get_observation(current_bars)
        observation = current_bars
        terminal = self._get_terminal()
        trade_amount = np.sum(np.abs(action[1:] - self.prev_action[1:]))
        # We do not calculate actual mu for speeding up
        if not is_training:
            mu = calculate_pv_after_commission(action,
                                               self.prev_action,
                                               self.commission_rate)
            returns = mu * (returns + 1.) - 1.
            cost = 1 - mu
        else:
            cost = 0
        self.prev_action = deepcopy(action)
        reward = np.sum(returns * action)
        info = {
            "reward": reward,
            "returns": returns,
            "cost": cost,
            'trade_amount': trade_amount
        }
        return observation, reward, terminal, info

    def _update_time(self):
        self.current_step += 1
        self.current_time = self.time_index[self.current_step]

    def _get_terminal(self):
        return self.current_time >= self.end

    def _get_bar(self):
        bar = self.data_val[self.current_step]
        return bar

    @property
    def action_dim(self):
        return self.num_stocks

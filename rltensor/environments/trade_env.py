from copy import deepcopy
from collections import defaultdict
import pandas as pd
import numpy as np

from rltensor.environments.core import Env
from rltensor.environments.utils import seconds2date


class TradeEnv(Env):
    def __init__(self, data, start=None, end=None,
                 add_cash=True, keys=["close", "open", "high"]):
        self.keys = keys
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
        for symbol in self.symbols:
            val = []
            for col in dfs.keys():
                val.append(dfs[col][[symbol]])
            val = np.concatenate(val, axis=1)
            _data[symbol] = pd.DataFrame(np.array(val), columns=dfs.keys(),
                                         index=self.time_index)
        self.set_trange(start, end)
        self.dfs = dfs
        self.data = _data
        self.num_stocks = len(self.symbols)
        self.current_time = self.start
        self.current_step = 0
        # Use for calculate return
        self.prev_bars = self._get_bar()

    def set_trange(self, start=None, end=None):
        if start is None:
            self.start = self.time_index[0]
        else:
            start = pd.Timestamp(start)
            self.start = min(start, self.time_index[0])
        if end is None:
            self.end = self.time_index[-1]
        else:
            end = pd.Timestamp(end)
            self.end = max(end, self.time_index[-1])

    def _reset(self):
        self.current_time = self.start
        self.current_step = 0
        self.prev_bars = self._get_bar()
        observation = self._get_observation(self.prev_bars)
        return observation

    def _step(self, action, is_training=True, *args, **kwargs):
        """
        current_bars = self._get_bar()
        returns = []
        for symbol in self.symbols:
            returns.append(current_bars[symbol]["close"] / self.prev_bars[symbol]["close"] - 1)
        """
        returns = np.ones(len(self.symbols))
        returns = np.array(returns)
        # Update bars
        # self.prev_bars = deepcopy(current_bars)
        # self._update_time()
        # observation = self._get_observation(current_bars)
        observation = 0
        terminal = self._get_terminal()
        reward = np.sum(returns * action)
        info = {}
        info["returns"] = returns
        return observation, reward, terminal, info

    def _get_observation(self, bars):
        observation = []
        for symbol in self.symbols:
            observation.append([bars[symbol][key] for key in self.keys])
        return np.array(observation)

    def _update_time(self):
        index = list(self.time_index).index(self.current_time)
        self.current_time = self.time_index[index + 1]
        self.current_step += 1

    def _get_terminal(self):
        return self.current_time >= self.end

    def _get_bar(self):
        bar = {}
        for symbol in self.symbols:
            # bar[symbol] = self.data[symbol].loc[self.current_time]
            bar[symbol] = {}
            for key in self.keys:
                bar[symbol][key] = 1.0
            """
            min_t = self.impute_data[symbol]["time_range"][0]
            max_t = self.impute_data[symbol]["time_range"][1]
            if (min_t <= self.current_time) and (max_t >= self.current_time):
                if self.current_time in self.data[symbol].index:
                    bar[symbol] = self.data[symbol].loc[self.current_time]
                else:
                    bar[symbol] = deepcopy(self.impute_bar[symbol])
            elif min_t > self.current_time:
                bar[symbol] = self.impute_data[symbol]["impute_val"][0]
            else:
                bar[symbol] = self.impute_data[symbol]["impute_val"][1]
            """
        # Keep value for imputation
        # self.impute_bar = deepcopy(bar)
        return bar

    @property
    def action_dim(self):
        return self.num_stocks

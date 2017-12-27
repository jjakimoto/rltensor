from copy import deepcopy
import pandas as pd
import numpy as np

# from .utils import convert_time
from .core import Env


class TradeEnv(Env):
    """Environment only for close prices"""

    def __init__(self, data, start=None, end=None,
                 add_cash=True, keys=["close", "open", "high"]):
        self.keys = keys
        time_index = set()
        impute_data = {}
        data = deepcopy(data)
        for key, val in data.items():
            dates = val["date"].values
            # dates = [convert_time(d) for d in dates]
            impute_data[key] = dict(time_range=(dates[-1], dates[0]),
                                    impute_val=(val.iloc[-1], val.iloc[0]))
            data[key].index = dates
            time_index = time_index.union(set(dates))
        self.time_index = sorted(list(time_index))
        if add_cash:
            val = np.ones(len(self.time_index))
            cash_df = pd.DataFrame({"open": val,
                                    "high": val,
                                    "low": val,
                                    "close": val,
                                    "volume": val},
                                   index=self.time_index)
            key = "Cash"
            data[key] = cash_df
            impute_data[key] = dict(time_range=(self.time_index[-1],
                                                self.time_index[0]),
                                    impute_val=(cash_df.iloc[-1],
                                                cash_df.iloc[0]))
        self.impute_data = impute_data
        if start is None:
            self.start = self.time_index[0]
        else:
            self.start = min(start, self.time_index[0])
        if end is None:
            self.end = self.time_index[-1]
        else:
            self.end = max(end, self.time_index[-1])
        self.data = data
        self.symbols = list(data.keys())
        self.num_stocks = len(self.symbols)
        self.current_time = self.start
        self.current_step = 0
        # Use for calculate return
        self.prev_bars = self._get_bar()
        self.max_time = max(self.time_index)

    def _reset(self):
        self.current_time = self.start
        self.current_step = 0
        self.prev_bars = self._get_bar()
        observation = self._get_observation(self.prev_bars)
        return observation

    def _step(self, action, is_training=True, *args, **kwargs):
        current_bars = self._get_bar()
        returns = []
        for symbol in self.symbols:
            returns.append(current_bars[symbol]["close"] / self.prev_bars[symbol]["close"] - 1)
        returns = np.array(returns)
        # Update bars
        self.prev_bars = deepcopy(current_bars)
        self._update_time()
        observation = self._get_observation(current_bars)
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
        index = self.time_index.index(self.current_time)
        self.current_time = self.time_index[index + 1]
        self.current_step += 1

    def _get_terminal(self):
        return self.current_time >= self.max_time

    def _get_bar(self):
        bar = {}
        for symbol in self.symbols:
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
        # Keep value for imputation
        self.impute_bar = deepcopy(bar)
        return bar

    @property
    def action_dim(self):
        return self.num_stocks

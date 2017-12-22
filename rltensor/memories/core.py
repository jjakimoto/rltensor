from collections import namedtuple, deque
from copy import deepcopy
import numpy as np


Experience = namedtuple('Experience',
                        'state0, action, reward, state1, terminal1')


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class BaseMemory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)
        self.ignore_episode_boundaries = ignore_episode_boundaries

    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, *args, **kwargs):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
        self.obs_shape = observation.shape

    def get_recent_state(self, observation=None):
        _observations = deepcopy(self.recent_observations)
        if observation is not None:
            _observations.append(observation)
        while len(_observations) < self.window_length:
            _observations.insert(0, np.zeros(self.obs_shape))
        # Make sure window length observations
        assert len(_observations) == self.window_length
        return np.array(_observations)

    def reset(self):
        self.recent_observations = deque(maxlen=self.window_length)
        self.recent_terminals = deque(maxlen=self.window_length)

    @property
    def nb_entries(self):
        return len(self.observations)

    def update_weights(self, *args, **kwargs):
        pass

    def add_weights(self):
        pass

    def get_weights(self):
        return None

    def get_importance_weights(self, batch_size=None):
        return None

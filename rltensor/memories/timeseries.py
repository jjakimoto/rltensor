import numpy as np
from collections import namedtuple, deque
from copy import deepcopy

from .core import BaseMemory, RingBuffer, zeroed_observation
from .utils import sample_batch_indexes


TSExperience = namedtuple('TSExperience',
                          'state, action, reward, terminal index')


class TSMemory(BaseMemory):
    def __init__(self, limit, window_length, beta=5.0e-5, *args, **kwargs):
        super(TSMemory, self).__init__(window_length=window_length,
                                       ignore_episode_boundaries=True)
        self.limit = limit
        # Do not use deque to implement the memory. This data structure
        # may seem convenient but
        # it is way too slow on random access. Instead, we use our own
        # ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self.recent_actions = None
        self.beta = beta

    def sample(self, batch_size, *args, **kwargs):
        num_options = self.nb_entries - (batch_size + self.window_length - 2)
        # Index has to be more than one at least for taking prev_action
        num_options -= 1
        assert num_options > 0
        idx = np.random.geometric(self.beta) % num_options
        init_idx = self.start_idx - idx
        batch_idxs = np.arange(init_idx - batch_size + 1, init_idx + 1)
        # Create experiences
        experiences = []
        # Each idx is index for state1
        for i, idx in enumerate(batch_idxs):
            # Observatio and terminal happens at the same time, so
            # previous index has to keep terminal==False.
            action = self.actions[idx - 1]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state = [self.observations[obs_i] for obs_i in range(idx - self.window_length + 1, idx + 1)]
            scale = np.array(state)[-1, :, 0]
            shape = np.array(state).shape
            scale = np.expand_dims(np.expand_dims(scale, axis=0), axis=-1)
            scale = np.tile(scale, [shape[0], 1, shape[2]])
            state = state / scale
            experiences.append(TSExperience(state=state, action=action,
                                            reward=reward, terminal=terminal,
                                            index=idx))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, info, is_store=True):
        super(TSMemory, self).append(observation, action, reward, terminal, info)
        # This needs to be understood as follows: in `observation`,
        # take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        self.recent_actions = deepcopy(action)
        if is_store:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(info["returns"])
            self.terminals.append(terminal)

    def _sample_batch_indexes(self, low, high, size, weights=None):
        return sample_batch_indexes(low, high, size, weights)

    @property
    def nb_entries(self):
        return len(self.observations)

    @property
    def start_idx(self):
        return self.nb_entries - 1

    def get_recent_actions(self):
        return self.recent_actions

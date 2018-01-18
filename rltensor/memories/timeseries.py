import numpy as np
from collections import namedtuple
from copy import deepcopy

from .core import BaseMemory, RingBuffer, zeroed_observation
from .utils import sample_batch_indexes


TSExperience = namedtuple('TSExperience',
                          'state, action, reward, terminal index')


class TSMemory(BaseMemory):
    def __init__(self, limit, window_length, beta=5.0e-5,
                 is_volume=False, *args, **kwargs):
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
        self.is_volume = is_volume

    def sample(self, batch_size, use_newest=False, *args, **kwargs):
        if use_newest:
            batch_idxs = np.arange(self.start_idx - batch_size + 1, self.start_idx + 1)
        else:
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
            history_idx = range(idx - self.window_length + 1, idx + 1)
            _observations = [self.observations[obs_i] for obs_i in history_idx]
            state = self._construct_state(_observations)
            experiences.append(TSExperience(state=state, action=action,
                                            reward=reward, terminal=terminal,
                                            index=idx))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, info, is_store=True):
        # This needs to be understood as follows: in `observation`,
        # take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        self.recent_actions = deepcopy(action)
        if is_store:
            self.observations.append(self.most_recent_observation)
            self.actions.append(action)
            self.rewards.append(info["returns"])
            self.terminals.append(terminal)
        super(TSMemory, self).append(observation, action, reward, terminal, info)

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

    def get_recent_state(self):
        x = np.concatenate((self.most_recent_observation['price'],
                            self.most_recent_observation['volume']),
                           axis=-1)
        observations = [self.most_recent_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            observations.insert(0, self.recent_observations[current_idx])
        while len(observations) < self.window_length:
            observations.insert(0, {'price': zeroed_observation(self.most_recent_observation['price']),
                                    'volume': zeroed_observation(self.most_recent_observation['volume'])})
        state = self._construct_state(observations)
        return state

    def _construct_state(self, observations):
        # Normalize price and volume seprately
        price_state = [observation['price'] for observation in observations]
        price_scale = np.array(price_state)[-1, :, 0]
        shape = np.array(price_state).shape
        price_scale = np.expand_dims(np.expand_dims(price_scale, axis=0),
                                     axis=-1)
        price_scale = np.tile(price_scale, [shape[0], 1, shape[2]])
        price_state = price_state / price_scale

        if self.is_volume:
            volume_state = [observation['volume'] for observation in observations]
            volume_scale = np.array(volume_state)[-1, :]
            volume_state = volume_state / volume_scale
            state = np.concatenate((price_state, volume_state), axis=-1)
        else:
            state = price_state
        return state

    def set_recent_data(self, observations, terminals, actions):
        super().set_recent_data(observations, terminals)
        self.recent_actions = deepcopy(actions)

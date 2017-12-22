import numpy as np
from six.moves import xrange

from .core import BaseMemory, Experience, RingBuffer
from .utils import sample_batch_indexes


class SequentialMemory(BaseMemory):
    def __init__(self, window_length, limit, *args, **kwargs):
        super(SequentialMemory, self).__init__(window_length)
        self.limit = limit
        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, weights=None, batch_idxs=None):
        if batch_idxs is None:
            if weights is not None:
                _weights = weights[1:]
                _weights /= np.sum(_weights)
            else:
                _weights = None
            # Draw random indexes such that we have at least a single entry before each
            # index. Thus, draw samples from [1, self.nb_entries)
            batch_idxs = self._sample_batch_indexes(1, self.nb_entries, batch_size, _weights)
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        if weights is not None:
            _weights = weights[:-1]
            _weights /= np.sum(_weights)
        else:
            _weights = None
        experiences = []
        # Each idx is index for state1
        for i, idx in enumerate(batch_idxs):
            # Observatio and terminal happens at the same time, so
            # previous index has to keep terminal==False.
            s0_i = idx - 1
            terminal0 = self.terminals[s0_i]
            while terminal0:
                # Repeat sampling until getting proper idx
                s0_i  = self._sample_batch_indexes(0, self.nb_entries-1, 1, _weights)[0]
                terminal0 = self.terminals[s0_i]
                batch_idxs[i] = s0_i + 1
            assert 0 <= s0_i < self.nb_entries - 1

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[s0_i],]
            for offset in xrange(1, self.window_length):
                current_idx = s0_i - offset
                current_terminal = self.terminals[current_idx] if current_idx >= 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            # Complete unobserved state with 0
            while len(state0) < self.window_length:
                state0.insert(0, np.zeros_like(state0[0]))
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal1 = self.terminals[idx]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        # Keep sampled sampled idx for prioritized sampling
        self.sampled_idx = batch_idxs
        return experiences

    def append(self, observation, action, reward, terminal, is_store=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal)
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if is_store:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    def _sample_batch_indexes(self, low, high, size, weights=None):
        return sample_batch_indexes(low, high, size, weights)

    @property
    def nb_entries(self):
        return len(self.observations)

from six.moves import xrange
import random
import warnings
import numpy as np

def sample_batch_indexes(low, high, size, weights=None):
    """Sample indexes from [low, high)"""
    assert low < high
    r = xrange(low, high)
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        if weights is None:
            batch_idxs = random.sample(r, size)
        else:
            batch_idxs = np.random.choice(r, size, False, weights)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.choice(r, size, True, weights)
    assert len(batch_idxs) == size
    return np.array(batch_idxs, dtype=int)

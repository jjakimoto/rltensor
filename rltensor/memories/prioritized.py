import numpy as np
from six.moves import xrange

from .sequential import SequentialMemory
from .utils import sample_batch_indexes


class PrioritizedMemory(SequentialMemory):
    def __init__(self, window_length, limit, alpha=0.5, beta=0.5, annealing_step=1e6, 
                 epsilon=1e-4, *args, **kwargs):
        super(PrioritizedMemory, self).__init__(window_length, limit, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.beta_init = beta
        self.annealing_step = annealing_step
        self.epsilon = epsilon
        self.priorities = []
        self.sampled_idx = None

    def update_weights(self, step, error=None, *args, **kwargs):
        self.priorities = np.ones(self.nb_entries) * self.epsilon
        self.beta = self._calc_beta(step)
        if error is not None:
            self.priorities[self.sampled_idx] = np.abs(error) + self.epsilon
            
    def add_weights(self):
        while len(self.priorities) < self.nb_entries:
            self.priorities = np.append(self.priorities, np.max(self.priorities))
        
    def get_weights(self):
        if self.priorities is not None:
            return None
        else:
            weights = self.priorities**self.alpha
            return weights / np.sum(weights)
        
    def get_importance_weights(self, batch_size=None):
        if self.priorities is None:
            return np.ones(batch_size)
        else:
            weights = self.priorities[self.sampled_idx] ** (-self.alpha*self.beta)
            return weights / np.max(weights)
    
    def _calc_beta(self, step):
        return self.beta_init + (1 - self.beta_init) * (self.annealing_step - step) / self.annealing_step
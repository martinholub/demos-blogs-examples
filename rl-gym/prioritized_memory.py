import random
import numpy as np
from SumTree import SumTree

class PERMemory(object):
    """Prioritized Experience Replay Memory

    https://arxiv.org/abs/1511.05952
    https://github.com/rlcode/per
    """
    # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_error = 1.
        # abs_err_upper = 1. # clipped abs error

    def _get_priority(self, error):
        if np.max(error) > self.max_error: self.max_error = np.max(error)
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def batch_update(self, idxs, errors):
        """update errors for bunch of samples from memory"""
        ps = self._get_priority(errors)
        for i, p in zip(idxs, ps):
            self.tree.update(i, p)

    def _shuffle_lists(*lists):
        """shuffle multiple lists together consistently
        # done by keras
        """
        combi_list = list(zip(*lists))
        random.shuffle(combi_list)
        return zip(*combi_list)

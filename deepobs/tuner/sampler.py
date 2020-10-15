# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>

import numpy as np
from scipy.stats.distributions import uniform, norm


class log_uniform():
    """A log uniform distribution that takes an arbitrary base."""
    def __init__(self, a, b, base=10):
        """
        Args:
            a (float): Lower bound.
            b (float): Range from lower bound.
            base (float): Base of the log.
        """
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform_values = uniform(loc=self.loc, scale=self.scale)
        exp_values = np.power(self.base, uniform_values.rvs(size=size, random_state=random_state))
        if len(exp_values) == 1:
            return exp_values[0]
        else:
            return exp_values


class negative_log_uniform(log_uniform):
    def __init__(self, a, b, base=10):
        super().__init__(a, b, base)
    
    def rvs(self, size=1, random_state=None):
        ret = super().rvs(size, random_state)
        return 1. - ret


class tuple_sampler:
    def __init__(self, *samplers):
        self.samplers = samplers
    
    def rvs(self, size=1, random_state=None):
        return tuple(x.rvs(size, random_state) for x in self.samplers)
    

class log_normal:
    """distribution where log(X) is normal"""
    def __init__(self, mu, sigma, base=10):
        self.mu = mu
        self.sigma = sigma
        self.base = base
    
    def rvs(self, size=1, random_state=None):
        normal = norm(loc=self.mu, scale=self.sigma)
        rand_values = np.power(self.base, normal.rvs(size=size, random_state=random_state))
        return rand_values[0] if len(rand_values) == 1 else rand_values


class negative_log_normal(log_normal):
    def __init__(self, mu, sigma, base=10):
        super().__init__(mu, sigma, base)
    
    def rvs(self, size=1, random_state=None):
        return 1. - super().rvs(size, random_state)
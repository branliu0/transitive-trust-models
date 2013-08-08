import numpy as np
from scipy import stats


def clamp(val, minval, maxval):
    """ Clamps a value such that it is between two values. """
    return min(max(val, minval), maxval)


def softmax_rv(masses, values=None, z=0.5):
    """ Returns a discrete random variable based on the softmax function. """
    nums = np.exp(np.array(masses) / z)
    dist = nums / np.sum(nums)
    if not values:
        values = np.arange(len(masses))
    return stats.rv_discrete(name='softmax', values=(values, dist))


def discrete_uniform_rv(values):
    """ Returns a discrete RV that one of the supplied values with unif prob. """
    n = len(values)
    dist = np.ones(n) / n
    return stats.rv_discrete(name='discrete_uniform', values=(values, dist))


def noisy_theta(prior_type, theta_i, theta_j):
    lower_bound = theta_j - ((1 - theta_i) / 2)
    if prior_type == 'uniform':
        val = stats.uniform.rvs(loc=lower_bound, scale=(1 - theta_i))
        return clamp(val, 0, 1)
    elif prior_type == 'normal':
        val = stats.norm.rvs()
    elif prior_type == 'beta':
    raise ValueError("Invalid prior type")


def random_true(prob):
    """ Returns True with probability prob or False otherwise. """
    return np.random.random() < prob


def resample_unique(rv, existing_values=[]):
    """ Samples a RV until we get a value unique from previous values. """
    while True:
        sample = rv.rvs()
        if sample not in existing_values:
            return sample

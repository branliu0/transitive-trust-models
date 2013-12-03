import math
import numpy as np
from scipy import stats


def clamp(val, minval, maxval):
    """ Clamps a value such that it is between two values. """
    return min(max(val, minval), maxval)


def normalize(array):
    """ Normalizes an array of values so that the sum of the array is unity. """
    total = float(sum(array))
    return [x / total for x in array]


def softmax_rv(masses, values=None, z=0.05):
    """ Returns a discrete random variable based on the softmax function. """
    nums = np.exp(np.array(masses) / z)
    dist = nums / np.sum(nums)
    if values is None:
        values = np.arange(len(masses))
    return stats.rv_discrete(name='softmax', values=(values, dist))


def discrete_uniform_rv(values):
    """ Returns a discrete RV that one of the supplied values with unif prob. """
    n = len(values)
    dist = np.ones(n) / n
    return stats.rv_discrete(name='discrete_uniform', values=(values, dist))


def agent_type_rv(prior_type):
    """ Returns the random variable corresponding to the agent prior type. """
    if prior_type == 'uniform':
        return stats.uniform()
    elif prior_type == 'normal':
        # [-sqrt(0.5), sqrt(0.5)] is the range of the truncnorm *before* scaling.
        # loc is the mean of the distribution
        # scale is the standard deviation of the distribution
        return stats.truncnorm(
            -math.sqrt(0.5), math.sqrt(0.5), loc=0.5, scale=math.sqrt(0.5))
    elif prior_type == 'beta':
        return stats.beta(2, 2)
    raise ValueError('Invalid agent type prior')


def noisy_theta_rv_args(prior_type, theta_i, theta_j):
    """ Returns the args and kwargs for constructing the noisy theta RV.

    Sadly, we need this because frozen rvs don't have the expect method, so
    we need to use these args directly to the expect method.
    """
    lower_bound = theta_j - 0.5 * (1 - theta_i)
    if prior_type == 'uniform':
        # loc is the lower bound for the uniform distribution
        return (), {'loc': lower_bound, 'scale': (1 - theta_i)}
    elif prior_type == 'normal':
        # [-sqrt(0.5), sqrt(0.5)] is the range of the truncnorm *before* scaling.
        # loc is the mean of the distribution: theta_j
        # scale is the standard deviation of the distribution
        return ((-math.sqrt(0.5), math.sqrt(0.5)),
                {'loc': theta_j, 'scale': math.sqrt(0.5) * (1 - theta_i)})
    elif prior_type == 'beta':
        # loc is the lower bound for the beta distribution
        # scale just scales the beta distribution
        return (2, 2), {'loc': lower_bound, 'scale': (1 - theta_i)}
    raise ValueError('Invalid agent type prior')

def noisy_theta_rv(prior_type, theta_i, theta_j):
    """ Returns the random variable object for the noisy theta """
    args, kwargs = noisy_theta_rv_args(prior_type, theta_i, theta_j)
    if prior_type == 'uniform':
        return stats.uniform(*args, **kwargs)
    elif prior_type == 'normal':
        return stats.truncnorm(*args, **kwargs)
    elif prior_type == 'beta':
        return stats.beta(*args, **kwargs)
    raise ValueError('Invalid prior type')


def noisy_theta(prior_type, theta_i, theta_j):
    """ Returns a single realization of the noisy theta RV, clamped to [0, 1] """
    val = noisy_theta_rv(prior_type, theta_i, theta_j).rvs()
    return clamp(val, 0, 1)


def expected_noisy_theta(prior_type, theta_i, theta_j):
    """ Returns the expected value of the 'noisy theta' random variable.

    E[X] = Pr[0 < X < 1] * E[X | 0 < X < 1] + Pr[X > 1]

    Note: This function can be a little bit slow because of the calculation
    of the expected value.
    """
    if prior_type == 'uniform':
        # We can do the uniform case manually, and do it much much faster.
        bottom = theta_j - 0.5 * (1 - theta_i)
        top = theta_j + 0.5 * (1 - theta_i)
        pr_center = (min(1, top) - max(0, bottom)) / (top - bottom)
        ex_center = 0.5 * (min(1, top) + max(0, bottom))
        pr_top = (max(1, top) - 1) / (top - bottom)
        return pr_center * ex_center + pr_top

    # For the truncnorm and beta case, we'll just defer to using the CDF, which
    # is unfortunately slower.
    args, kwargs = noisy_theta_rv_args(prior_type, theta_i, theta_j)
    rv = noisy_theta_rv(prior_type, theta_i, theta_j)

    pr_center = rv.cdf(1) - rv.cdf(0)  # Pr[0 < X < 1]
    # Sadly, the expect function is not connected to frozen random variables.
    kwargs['lb'] = 0  # Lower bound
    kwargs['ub'] = 1  # Upper bound
    kwargs['conditional'] = True  # Do a conditional expectation
    kwargs['epsabs'] = 1e-3  # Lower the error tolerance so it integrates faster
    ex_center = rv.dist.expect(args=args, **kwargs)  # E[X | 0 < X < 1]
    pr_top = 1 - rv.cdf(1)  # Pr[X > 1]

    return pr_center * ex_center + pr_top


def random_true(prob):
    """ Returns True with probability prob or False otherwise. """
    return np.random.random() < prob


def resample_unique(rv, existing_values=[]):
    """ Samples a RV until we get a value unique from previous values. """
    while True:
        sample = rv.rvs()
        if sample not in existing_values:
            return sample


class RegenList(list):
    """ Regenerating List, convenient for auto-regenerating values in bulk

    This slight extension to Python's native list data structure makes it easy
    to create pseudo-streams that continually provide values, but generate
    them in bulk (for computational efficiency).
    """

    def __init__(self, gen_lambda, *args):
        """
        Args:
            gen_lambda: A function that returns a list, which generates the
                the values for this list.
            args: Any functions to be passed to gen_lambda.
        """
        super(RegenList, self).__init__()
        self.gen_lambda = gen_lambda
        self.args = args
        self.extend(gen_lambda(*self.args))

    def pop(self, *args, **kwargs):
        if not self:
            self.regen(*self.args)
        return super(RegenList, self).pop(*args, **kwargs)

    def shift(self):
        """ Sadly Python's list does not have a shift function. """
        return self.pop(0)

    def regen(self, *args):
        self.extend(self.gen_lambda(*args))


class RandomRegenList(RegenList):

    def __init__(self, N):
        super(RandomRegenList, self).__init__(lambda: np.random.random(size=N))


class CoinFlipRegenList(RegenList):

    def __init__(self, N, p):
        super(CoinFlipRegenList, self).__init__(
            lambda: np.random.random(size=N) < p)

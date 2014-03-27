import math
import random
from collections import deque

import networkx as nx
import numpy as np
from scipy import stats


def clamp(val, minval, maxval):
    """ Clamps a value such that it is between two values. """
    return min(max(val, minval), maxval)


def normalize(array):
    """ Normalizes an array of values so that the sum of the array is unity. """
    total = float(sum(array))
    return [x / total for x in array]


def random_round(i):
    base = int(i)
    dec = i - base
    return base + (1 if random.random() > dec else 0)


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


def random_weighted_graph(num_nodes, edge_prob, weight_dist='uniform'):
    """ Returns a Erdos-Renyi graph with edge weights. """
    g = nx.gnp_random_graph(num_nodes, edge_prob, directed=True)
    weights = agent_type_rv(weight_dist).rvs(size=g.number_of_edges())
    for i, e in enumerate(g.edges_iter()):
        g[e[0]][e[1]]['weight'] = weights[i]
    return g

def weighted_ba_graph(N, K, weight_dist='uniform'):
    g = nx.barabasi_albert_graph(N, K)
    weights = agent_type_rv(weight_dist).rvs(size=g.number_of_edges())
    for i, e in enumerate(g.edges_iter()):
        g[e[0]][e[1]]['weight'] = weights[i]
    return g


def gauss_jordan(m, eps = 1.0/(10**10)):
    """ Puts given matrix (2D array) into the Reduced Row Echelon Form.

    Returns True if successful, False if 'm' is singular.  NOTE: make sure all
    the matrix items support fractions! Int matrix will NOT work!  Written by
    Jarno Elonen in April 2005, released into Public Domain.
    """

    (h, w) = (len(m), len(m[0]))
    for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:     # Singular?
            return False
        for y2 in range(y+1, h):    # Eliminate column y
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]
        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
    for x in range(h, w):       # Normalize row y
        m[y][x] /= c
    return True


def ls_solve(M, b):
    """
    solves M*x = b
    return vector x so that M*x = b
    :param M: a matrix in the form of a list of list
    :param b: a vector in the form of a simple list of scalars
    """
    m2 = [row[:]+[right] for row,right in zip(M,b) ]
    return [row[-1] for row in m2] if gauss_jordan(m2) else None


def gt_graph_from_nx(graph):
    import graph_tool.all as gt
    g = gt.Graph()
    v = list(g.add_vertex(len(graph)))
    weights = g.new_edge_property("double")
    for i, j, d in graph.edges(data=True):
        e = g.add_edge(v[i], v[j])
        weights[e] = d['weight']
    g.edge_properties['weight'] = weights
    return g


def fast_max_flow(gt_graph, i, j):
    import graph_tool.all as gt
    cap = gt_graph.edge_properties['weight']
    res = gt.push_relabel_max_flow(
        gt_graph, gt_graph.vertex(i), gt_graph.vertex(j), cap)
    return sum(cap[e] - res[e] for e in gt_graph.vertex(j).in_edges())


class RegenList(deque):
    """ Regenerating List, convenient for auto-regenerating values in bulk

    Uses python's deque data structure for high performance.
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

    def pop(self):
        if not self:
            self.regen(*self.args)
        return self.popleft()

    def regen(self, *args):
        self.extend(self.gen_lambda(*args))


class RandomRegenList(RegenList):

    def __init__(self, N):
        super(RandomRegenList, self).__init__(lambda: np.random.random(size=N))


class CoinFlipRegenList(RegenList):

    def __init__(self, N, p):
        super(CoinFlipRegenList, self).__init__(
            lambda: np.random.random(size=N) < p)


class RandomWalk(object):

    def __init__(self, graph, alpha):
        N = graph.number_of_nodes()
        self.terminator = CoinFlipRegenList(int(N * N / alpha), alpha)
        self.steps = {}
        self.rvs = {}
        for node in graph.nodes():
            edges = graph.edges(node, data=True)
            if edges:
                values = np.array([x[1] for x in edges])
                probs = normalize([x[2]['weight'] for x in edges])
                bins = np.cumsum(probs)
                self.steps[node] = RegenList(
                    lambda values, bins: values[np.digitize(
                        np.random.random(int(N / alpha)), bins)], values, bins)
            else:
                # What else can be done for dangling nodes?
                self.steps[node] = RegenList(lambda: [None] * int(N / alpha))

    def terminates(self):
        return self.terminator.pop()

    def step(self, node):
        return self.steps[node].pop()

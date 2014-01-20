"""This module deals with with 4 transitive trust models:
    1. PageRank
    2. Hitting Time
    3. Maximum Flow
    4. Shortest Path
Please refer to the individual functions for more details on the trust
models.
"""
import heapq
import random
import sys

import networkx as nx
import numpy as np
from scipy import stats

import utils

# This is the restart probability, i.e., the probability that at each step
# of a random walk, that the walk will restart. This value is used in
# PageRank, Hitting Time, and Hitting Time PageRank.
ALPHA = 0.15


def pagerank(graph, weighted=True):
    """ Pagerank algorithm with beta = 0.85.

    If unweighted, then every outgoing edge is considered uniformly.
    Otherwise, outgoing edges are weighted by their given weights.

    Returns:
        An array where the ith element corresponds to the pagerank score
        of agent i in the trust graph.
    """
    if weighted:
        return np.array(nx.pagerank_numpy(graph).values())
    else:
        return np.array(nx.pagerank_numpy(graph, weight=None).values())


def personalized_pagerank(graph, weighted=True):
    """ Personalized PageRank algorithm with beta=0.85.

    Returns:
        An NxN numpy array.
    """
    N = graph.number_of_nodes()
    weight = 'weight' if weighted else None
    pagerank = np.zeros((N, N))
    personalization = {n: 0 for n in graph.nodes()}
    for i in xrange(N):
        personalization[i] = 1
        pagerank[i] = nx.pagerank_numpy(graph, personalization=personalization,
                                        weight=weight).values()
        personalization[i] = 0
    return pagerank


def _hitting_time_pretrusted_set(graph, pretrust_strategy):
    """ Pre-trusted set of nodes for Hitting Time, given a strategy.

    Args:
        pretrust_strategy:
            'all': Takes the entire set of nodes as the pre-trusted set.
            'prob': Probabilistically adds agent i to the pre-trusted set
                with probability theta_i.
            'top': Takes the top NUM_TOP_AGENTS agents by type as the
                pre-trusted set.
    Returns:
        A set of nodes from the graph, representing the pre-trusted set.
    """
    NUM_TOP_AGENTS = 10

    if pretrust_strategy == 'all':
        return set(graph.nodes())
    elif pretrust_strategy == 'prob':
        pretrust_set = set()
        for node, attrs in graph.nodes_iter(data=True):
            if utils.random_true(attrs['agent_type']):
                pretrust_set.add(node)
        return pretrust_set
    elif pretrust_strategy == 'top':
        top_nodes = heapq.nlargest(
            NUM_TOP_AGENTS, graph.nodes(data=True),
            key=lambda x: x[1]['agent_type'])
        return set(x[0] for x in top_nodes)
    raise ValueError("Invalid pretrust set strategy")


def _hitting_time_single(graph, target_node, pretrust_set, weighted=True):
    """ Returns the hitting time for a single node. """
    # TODO: Can tune these numbers to get to a reasonable epsilon
    MIN_ITERS = 3000
    MAX_ITERS = 5000
    MIN_HITS = 2

    num_nodes = graph.number_of_nodes()

    # Pregenerate a larger number of walks at once to try to save time.
    def generate_walks(node, size=MIN_ITERS/10):
        edges = graph.edges(node, data=True)
        if not edges:
            return [node] * size
        if weighted:
            rv = stats.rv_discrete(
                values=([x[1] for x in edges],
                            utils.normalize([x[2]['weight'] for x in edges])))
            return list(rv.rvs(size=size))
        else:
            neighbors = [x[1] for x in edges]
            return [random.choice(neighbors) for _ in xrange(size)]

    walks = [[] for i in xrange(num_nodes)]
    num_hits = 0
    num_steps = 0
    num_iters = 0

    # Actually run the Monte-Carlo simulations

    while (num_iters < MIN_ITERS or
            (num_hits < MIN_HITS and num_iters < MAX_ITERS)):
        cur_node = random.sample(pretrust_set, 1)[0]
        num_iters += 1
        while True:
            # We hit the target!
            if cur_node == target_node:
                num_hits += 1
                break

            if utils.random_true(ALPHA):
                break  # We jumped; start next iteration
            else:
                num_steps += 1
                try:
                    cur_node = walks[cur_node].pop()
                except IndexError:
                    walks[cur_node] = generate_walks(cur_node)
                    cur_node = walks[cur_node].pop()

    # print "%d steps taken" % num_steps
    # print "%d hits" % num_hits
    sys.stdout.write('.')  # Just to give an indicator of progress

    return float(num_hits) / num_iters


def _hitting_time_all(graph, pretrust_set, weighted=True, num_iters=1e5):
    def generate_walks(node, size=100):
        edges = graph.edges(node, data=True)
        if not edges:
            return [node] * size
        if weighted:
            rv = stats.rv_discrete(
                values=([x[1] for x in edges],
                            utils.normalize([x[2]['weight'] for x in edges])))
            return list(rv.rvs(size=size))
        else:
            neighbors = [x[1] for x in edges]
            return [random.choice(neighbors) for _ in xrange(size)]

    def generate_coin_flips(size):
        return list(stats.bernoulli.rvs(ALPHA, size=size))

    num_nodes = graph.number_of_nodes()
    hits = np.zeros(num_nodes)
    coin_flips = generate_coin_flips(8 * num_iters)
    walks = [generate_walks(i) for i in xrange(num_nodes)]

    for i in xrange(num_iters):
        if i % (num_iters / 50) == 0:
            sys.stdout.write('.')

        cur_node = random.sample(pretrust_set, 1)[0]
        current_hits = dict.fromkeys(graph.nodes(), 0)
        while True:
            current_hits[cur_node] = 1  # Mark as hit, but only once
            if coin_flips.pop():  # Restart?
                break

            if not coin_flips:
                coin_flips = generate_coin_flips(1000)

            if not walks[cur_node]:
                walks[cur_node] = generate_walks(cur_node)

            cur_node = walks[cur_node].pop()

        for n in graph.nodes():
            hits[n] += current_hits[n]

    print
    return list(hits / num_iters)

def hitting_time(graph, pretrust_strategy, weighted=True, num_iters=1e5):
    """ Hitting Time algorithm

    The Hitting Time of a node is the probability that a random walk
    started from a pretrusted set hits the node before jumping/restarting.

    Args:
        pretrust_strategy: 'all', 'prob', or 'top'. See
            _hitting_time_pretrusted_set for more details.
        weighted: Boolean for whether the HT algorithm should follow edges
            based on weights or irrespective of their weights.

    Returns:
        An array where the ith element corresponds to the hitting time
        of agent i.
    """
    pretrust_set = _hitting_time_pretrusted_set(graph, pretrust_strategy)
    return _hitting_time_all(pretrust_set, weighted, num_iters)

def hitting_pagerank(graph, pretrust_strategy):
    """ Uses an eigenvector method to compute hitting time.

    Args:
        pretrust_strategy: 'all', 'prob', or 'top'. See
            _hitting_time_pretrusted_set for more details.

    Returns:
        An array where the ith element corresponds to the hitting time
        of agent i.
    """
    pretrust_set = _hitting_time_pretrusted_set(graph, pretrust_strategy)
    num_nodes = graph.number_of_nodes()

    # Restart distribution: Uniform across all pretrusted nodes
    restart = np.zeros(num_nodes)
    restart[list(pretrust_set)] = 1.0/len(pretrust_set)

    scores = []
    for i in xrange(num_nodes):
        adj_matrix = nx.to_numpy_matrix(graph)

        # Delete the out-nodes for node i, replace them with outedges
        # that go straight back to the restart distribution, thus
        # simulating the end of a "hit" by returning to the restart
        # distribution.
        adj_matrix[i] = restart

        # For dangling nodes, we add a self-edge, which simulates getting
        # "stuck" until we teleport.
        for j in xrange(num_nodes):
            if adj_matrix[j].sum() == 0:
                adj_matrix[j, j] = 1

        # Normalize outgoing edge weights for all nodes.
        for j in xrange(num_nodes):
            adj_matrix[j] /= adj_matrix[j].sum()

        # Now add in the restart distribution to the matrix.
        htpr_matrix = (1 - ALPHA) * adj_matrix + \
                ALPHA * np.outer(np.ones(num_nodes), restart)

        # To obtain PageRank score, take the dominant eigenvector, and
        # pull out the score for node i
        eigenvalues, eigenvectors = np.linalg.eig(htpr_matrix.T)
        dominant_index = eigenvalues.argsort()[-1]
        pagerank = np.array(eigenvectors[:, dominant_index]).flatten().real
        pagerank /= np.sum(pagerank)

        # Using Theorem 1 equation (ii) from Sheldon & Hopcroft 2007 and
        # using the fact that for node i, the expected return time is just
        # one more than the expected hitting time, since the first step
        # away from node i will always be to a node in the pretrusted set,
        # we arrive at this equation for deriving hitting time.
        scores.append(1.0 / (1 - ALPHA + ALPHA / pagerank[i]))

    return scores

def max_flow(graph):
    """ All-pairs maximum flow.

    Uses Ford-Fulkerson with the Edmonds-Karp-Dinitz path selection rule to
    guarantee a running time of O(VE^2).

    Returns:
        An array of n arrays of length n, representing the maximum flow
        that can be pushed to each of the n nodes. None is used for pairs
        where 0 flow can be pushed.
    """
    num_nodes = graph.number_of_nodes()
    scores = np.zeros((num_nodes, num_nodes))
    for i in xrange(num_nodes):
        for j in xrange(num_nodes):
            if i == j:
                scores[i][j] = None
            else:
                mf = nx.max_flow(graph, i, j, capacity='weight')
                scores[i][j] = None if mf == 0 else mf
        sys.stdout.write('.')
    sys.stdout.write("\n")
    return scores

def shortest_path(graph):
    """ All-pairs shortest path on the graph of inverse weights.

    For each pair, calculates the sum of the weights on the shortest path
    (by weight) between each pair in the graph. Uses the inverse weights
    to calculate, and inverts the final weight, such that a higher score is
    considered better.

    Returns:
        An array of n arrays of length n, representing the weight of the
        shortest path between each pair of nodes on the graph of inverse
        weights. None is used for pairs where there is no path.
    """
    num_nodes = graph.number_of_nodes()
    nx_dict = nx.all_pairs_dijkstra_path_length(
        graph, weight='inv_weight')
    # Convert from dict format to array format
    shortest_paths = np.zeros((num_nodes, num_nodes))
    for i, d in nx_dict.iteritems():
        for j in xrange(num_nodes):
            try:
                shortest_paths[i][j] = 1.0 / d[j]
            except (KeyError, ZeroDivisionError):
                shortest_paths[i][j] = None
    return shortest_paths

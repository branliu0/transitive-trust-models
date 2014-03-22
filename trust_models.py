"""This module deals with with 4 transitive trust models:
    1. PageRank
    2. Hitting Time
    3. Maximum Flow
    4. Shortest Path
Please refer to the individual functions for more details on the trust
models.
"""
import heapq
import sys

import networkx as nx
import numpy as np

from hitting_time.mat_hitting_time import global_eigen_prob_ht
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
    restart[list(pretrust_set)] = 1

    return global_eigen_prob_ht(graph, restart, ALPHA)


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

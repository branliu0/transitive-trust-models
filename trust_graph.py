import networkx as nx
import numpy as np
from scipy import stats

import utils

class TrustGraph(nx.DiGraph):
    """ A networkx.DiGraph object that represents a trust graph.

    In particular, this graph is guaranteed to have a float in [0, 1] set as
    the 'agent_type' attribute for every node. Every directed edge (i, j) is
    guaranteed to have a float in [0, 1] for the 'weight' attribute representing
    the subjective trust of agent i for agent j. Every edge also has an
    'inv_weight' attribute which is the reciprocal of the 'weight' attribute
    for the convenience of the Shortest Path transitive trust model.
    """

    AGENT_TYPE_PRIORS = ['uniform', 'normal', 'beta']
    EDGE_STRATEGIES = ['uniform', 'cluster']
    EDGE_WEIGHT_STRATEGIES = ['sample', 'noisy', 'prior']

    def __init__(self, num_nodes, agent_type_prior, edge_strategy,
                     edges_per_node, edge_weight_strategy, num_weight_samples):
        """
        Args:
            num_nodes: Number of nodes in this graph.
            agent_type_prior:
                'uniform': Selected from Unif[0, 1]
                'normal': Selected from Normal[0.5, 1] truncated to [0, 1]
                'beta': Selected from Beta[2, 2]
            edge_strategy:
                'uniform': Neighbors are uniformly selected
                'cluster': High types are more likely to connect to high types
            edges_per_node: The number of outgoing edges each node has.
            edge_weight_strategy:
                'sample': Sample from true agent type
                'noisy': Low types more likely to sample from Bernoulli[0.5]
                'prior': Low types more likely to sampel from prior distribution
            num_weight_samples: Number of times to sample for determining
                edge weights.
        Returns:
            A fully initialized TrustGraph object, generated using the
            parameters specified.
        """
        super(TrustGraph, self).__init__()

        MAX_INV_WEIGHT = 100000  # Maximum value for 'inv_weight'

        if agent_type_prior not in self.AGENT_TYPE_PRIORS:
            raise ValueError(
                "%s is an invalid agent type prior" % agent_type_prior)
        if edge_strategy not in self.EDGE_STRATEGIES:
            raise ValueError("%s is an invalid edge strategy" % edge_strategy)
        if edge_weight_strategy not in self.EDGE_WEIGHT_STRATEGIES:
            raise ValueError(
                "%s is an invalid edge weight strategy" % edge_weight_strategy)

        self.num_nodes            = int(num_nodes)
        self.agent_type_prior     = agent_type_prior
        self.edge_strategy        = edge_strategy
        self.edges_per_node       = int(edges_per_node)
        self.edge_weight_strategy = edge_weight_strategy
        self.num_weight_samples   = num_weight_samples

        # First generate our agent types, edges, and edge weights
        self.agent_types = TrustGraph.initialize_agent_types(
            self.num_nodes, self.agent_type_prior)
        edges = TrustGraph.initialize_edges(
            self.agent_types, self.edge_strategy, self.edges_per_node)
        edge_weights = TrustGraph.initialize_edge_weights(
            self.agent_types, edges, self.edge_weight_strategy,
            self.agent_type_prior, self.num_weight_samples)

        # Now let's add them to the networkx graph
        for i, agent_type in enumerate(self.agent_types):
            self.add_node(i, agent_type=agent_type)
        for i, neighbors in enumerate(edge_weights):
            for j, weight in enumerate(neighbors):
                if weight is not None:
                    inv_weight = (MAX_INV_WEIGHT if weight == 0.0
                                  else min(1.0 / weight, MAX_INV_WEIGHT))
                    self.add_edge(i, j, weight=weight, inv_weight=inv_weight)

        self.gt_graph = utils.gt_graph_from_nx(self)

    @staticmethod
    def initialize_agent_types(num_nodes, agent_type_prior):
        """
        Args:
            num_nodes: Number of agents.
            agent_type_prior: 'uniform', 'normal', or 'beta'. See docs for
                TrustGraph.create_graph.
        Returns:
            An array of length num_nodes with floats in [0, 1] repesenting
            the agent types.
        """
        return sorted(utils.agent_type_rv(agent_type_prior).rvs(size=num_nodes))

    @staticmethod
    def initialize_edges(agent_types, edge_strategy, edges_per_node):
        # TODO: 'cluster' strategy can be optimized.
        """
        Args:
            agent_types: Array of floats in [0, 1] representing agent types.
            edge_strategy: 'uniform' or 'cluster'. See docs for
                TrustGraph.create_graph.
            edges_per_node: The number of outgoing edges per node.
        Returns:
            An array of arrays of integers $a$, where $j \in a[i]$ indicates
            an outgoing edge from $i$ to $j$. The arrays are zero-based. This
            is an adjacency list.
        """
        edges = []
        n = len(agent_types)
        if edge_strategy == 'uniform':
            # Uniformly pick a subset from all possible nodes.
            for i in xrange(len(agent_types)):
                edges.append(np.random.choice(
                    np.delete(np.arange(n), i), edges_per_node,
                    replace=False))
            return edges
        elif edge_strategy == 'cluster':
            # High types are more likely to be connected to other high types
            for i, agent_type in enumerate(agent_types):
                neighbors = []
                for j in xrange(edges_per_node):
                    if utils.random_true(agent_type):
                        # with prob theta_i, be more likely to select a high-type agent
                        softmax = utils.softmax_rv(
                            np.delete(agent_types, neighbors + [i]),
                            np.delete(np.arange(n), neighbors + [i]))
                        neighbors.append(softmax.rvs())
                    else:
                        # with prob (1 - theta_i), sample uniformly from neighbors
                        uniform = utils.discrete_uniform_rv(
                            np.delete(np.arange(n), neighbors + [i]))
                        neighbors.append(uniform.rvs())
                edges.append(neighbors)
            return edges
        raise ValueError("Invalid edge strategy")

    @staticmethod
    def _expected_edge_weight(agent_types, edge_weight_strategy,
                              agent_type_prior, i, j):
        theta_i, theta_j = agent_types[i], agent_types[j]
        if edge_weight_strategy == 'sample':
            return theta_j
        elif edge_weight_strategy == 'noisy':
            return (1 - theta_i) * 0.5 + theta_i * theta_j
        elif edge_weight_strategy == 'prior':
            return (1 - theta_i) * utils.expected_noisy_theta(
                agent_type_prior, theta_i, theta_j) + theta_i * theta_j
        raise ValueError("Invalid edge weight strategy")

    @staticmethod
    def _sampled_edge_weight(agent_types, edge_weight_strategy,
                             agent_type_prior, num_samples, i, j):
        theta_i, theta_j = agent_types[i], agent_types[j]
        if edge_weight_strategy == 'sample':
            return stats.binom.rvs(num_samples, theta_j) / float(num_samples)
        elif edge_weight_strategy == 'noisy':
            # First find number of "true" samples
            true_samples = stats.binom.rvs(num_samples, theta_i)
            # Then sample from Bern[theta_j] `true_samples` times and from
            # Bern[0.5] (num_samples - true_samples) times.
            return (stats.binom.rvs(true_samples, theta_j) +
                    stats.binom.rvs(num_samples - true_samples, 0.5)) / \
                    float(num_samples)
        elif edge_weight_strategy == 'prior':
            true_samples = stats.binom.rvs(num_samples, theta_i)
            return (stats.binom.rvs(true_samples, theta_j) +
                    sum(stats.bernoulli.rvs(utils.noisy_theta(
                        agent_type_prior, theta_i, theta_j))
                        for _ in xrange(num_samples - true_samples))) / \
                    float(num_samples)
        raise ValueError("Invalid edge weight strategy")

    @staticmethod
    def initialize_edge_weights(agent_types, edges, edge_weight_strategy,
                                agent_type_prior, num_samples):
        """
        Args:
            agent_types: A list of floats in [0, 1] representing the agent types.
            edges: A list of adjacency lists representing the edges.
            edge_weight_strategy: 'sample', 'noisy', or 'prior'
            agent_type_prior: 'uniform', 'normal', or 'beta'
            num_samples: The number of samples used for defining the weights.
        Returns:
            An adjacency matrix of edge weights. None specifies the absence
            of an edge. Otherwise the matrix entry contains the weight of that
            edge.
        """
        def expected_weight(i, j):
            return TrustGraph._expected_edge_weight(
                agent_types, edge_weight_strategy, agent_type_prior, i, j)

        def sampled_weight(i, j):
            return TrustGraph._sampled_edge_weight(
                agent_types, edge_weight_strategy, agent_type_prior,
                num_samples, i, j)

        n = len(agent_types)
        weights = np.repeat(None, n * n).reshape(n, n)

        for i in xrange(n):
            for j in edges[i]:
                if num_samples == np.inf:
                    weights[i][j] = expected_weight(i, j)
                else:
                    weights[i][j] = sampled_weight(i, j)
        return weights

    @staticmethod
    def g50():
        """ Convenience method for quickly getting a graph of 50 nodes. """
        return TrustGraph(50, 'uniform', 'uniform', 15, 'noisy', 30)

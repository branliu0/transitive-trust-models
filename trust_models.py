import heapq
import networkx as nx
from scipy import stats
import utils

class TrustModels(object):
    """ The class used to encapsulate logic related to transitive trust models.

    This class deals with with 4 transitive trust models:
        1. PageRank
        2. Hitting Time
        3. MaxFlow
        4. Shortest Path
    Please refer to the individual functions for more details on the trust
    models.
    """

    def __init__(self, graph):
        """
        Args:
            graph: A networkx.DiGraph object. Every node is expected to have
                a 'agent_type' attribute, and every edge is expected to have
                both a 'weight' attribute and 'inv_weight' attribute.
        """
        self.graph = graph

    def pagerank(self):
        """ Unweighted pagerank algorithm with beta = 0.85.

        Every outgoing edge is considered uniformly -- the weight is not
        considered.

        Returns:
            An array where the ith element corresponds to the pagerank score
            of agent i in the trust graph.
        """
        return nx.pagerank_numpy(self.graph, weight=None).values()

    def pagerank_weighted(self):
        """ Weighted PageRank algorithm with beta = 0.85.

        Outgoing edges are weighted by their given weights.

        Returns:
            An array where the ith element corresponds to the pagerank score
            of agent i in the trust graph.
        """
        return nx.pagerank_numpy(self.graph).values()

    def _hitting_time_pretrusted_set(self, pretrust_strategy):
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
            return set(self.graph.nodes())
        elif pretrust_strategy == 'prob':
            pretrust_set = set()
            for node, attrs in self.graph.nodes_iter(data=True):
                if utils.random_true(attrs['agent_type']):
                    pretrust_set.add(node)
            return pretrust_set
        elif pretrust_strategy == 'top':
            top_nodes = heapq.nlargest(
                NUM_TOP_AGENTS, self.graph.nodes(data=True),
                key=lambda x: x[1]['agent_type'])
            return set(x[0] for x in top_nodes)
        raise ValueError("Invalid pretrust set strategy")

    def _hitting_time_for_pair(self, i, j, pretrust_set, weighted):
        RESTART_PROB = 0.15
        NUM_ITERS = 1000

        num_hits = 0  # counter for number of times we hit target j

        # Initialize discrete random variables for the random walk
        # Note: This gets re-initialized for every pair, and could be optimized.
        neighbor_dists = []
        if weighted:
            for i in xrange(self.graph.num_nodes):
                edges = self.graph.edges(i, data=True)
                neighbor_dists[i] = stats.rv_discrete(
                    values=([x[1] for x in edges],
                             utils.normalize([x[2]['weight'] for x in edges])))
        else:
            for i in xrange(self.graph.num_nodes):
                neighbor_dists[i] = utils.discrete_uniform(
                    [x[1] for x in self.graph.edges(i)])

        # Actually run the Monte-Carlo simulations
        cur_node = i
        for _ in xrange(NUM_ITERS):
            while True:
                # We hit the target!
                if cur_node == j:
                    num_hits += 1
                    break

                if utils.random_true(RESTART_PROB):
                    break  # We jumped; start next iteration
                else:
                    cur_node = neighbor_dists[cur_node].rvs()

        return float(num_hits) / NUM_ITERS


    def hitting_time(self, pretrust_strategy, weighted):
        """ Hitting Time algorithm with beta = 0.85.

        The Hitting Time of a node is the probability that

        Every outgoing edge is considered uniformly.

        Args:
            pretrust_strategy: 'all', 'prob', or 'top'. See
                _hitting_time_pretrusted_set for more details.
            weighted: Boolean for whether the HT algorithm should follow edges
                based on weights or irrespective of their weights.

        Returns:
            An array where the ith element corresponds to the hitting time
            of agent i.
        """
        pretrust_set = self._hitting_time_pretrusted_set(pretrust_strategy)
        ht_scores = []
        for i in xrange(self.graph.num_nodes):
            neighbor_scores = []
            for j in xrange(self.graph.num_nodes):
                if i == j:
                    continue
                neighbor_scores.append(
                    self._hitting_time_for_pair(i, j, pretrust_set,
                                                weighted=weighted))
            ht_scores.append(neighbor_scores)
        return ht_scores

    def max_flow(self):
        """ All-pairs maximum flow.

        Uses Ford-Fulkerson with the Edmonds-Karp-Dinitz path selection rule to
        guarantee a running time of O(VE^2).

        Returns:
            An array of n arrays of length n - 1, representing the maximum flow
            that can be pushed to each of the n - 1 neighbors for each of the n
            nodes.
        """
        max_flow_scores = []
        for i in xrange(self.graph.num_nodes):
            neighbor_scores = []
            for j in xrange(self.graph.num_nodes):
                if i == j:
                    continue
                neighbor_scores.append(
                    nx.max_flow(self.graph, i, j, capacity='weight'))
            max_flow_scores.append(neighbor_scores)
        return max_flow_scores

    def shortest_path(self):
        """ All-pairs shortest path on the graph of inverse weights.

        For each pair, calculates the sum of the weights on the shortest path
        (by weight) between each pair in the graph. Uses the inverse weights
        to calculate.

        Returns:
            An array of n arrays of length n -1, representing the weight of the
            shortest path between each pair of nodes on the graph of inverse
            weights.
        """
        nx_dict = nx.all_pairs_dijkstra_path_length(
            self.graph, weight='inv_weight')
        # Convert from dict format to array format
        shortest_paths = [x.values() for x in nx_dict.values()]
        # Need to delete the diagonal values
        for i in xrange(self.graph.num_nodes):
            del shortest_paths[i][i]
        return shortest_paths

import heapq
import sys
import random

import networkx as nx
from scipy import stats

import utils

class TrustModels(object):
    """ The class used to encapsulate logic related to transitive trust models.

    This class deals with with 4 transitive trust models:
        1. PageRank
        2. Hitting Time
        3. Maximum Flow
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

    def pagerank(self, weighted):
        """ Pagerank algorithm with beta = 0.85.

        If unweighted, then every outgoing edge is considered uniformly.
        Otherwise, outgonig edges are weighted by their given weights.

        Returns:
            An array where the ith element corresponds to the pagerank score
            of agent i in the trust graph.
        """
        if weighted:
            return nx.pagerank_numpy(self.graph).values()
        else:
            return nx.pagerank_numpy(self.graph, weight=None).values()

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

    def _hitting_time_single(self, target_node, pretrust_set, weighted):
        """ Returns the hitting time for a single node. """
        RESTART_PROB = 0.15

        # TODO: Can tune these numbers to get to a reasonable epsilon
        MIN_ITERS = 200
        MAX_ITERS = 500
        MIN_HITS = 2

        # Pregenerate a larger number of walks at once to try to save time.
        def generate_walks(node, size=MIN_ITERS/10):
            if weighted:
                edges = self.graph.edges(node, data=True)
                rv = stats.rv_discrete(
                    values=([x[1] for x in edges],
                             utils.normalize([x[2]['weight'] for x in edges])))
                return list(rv.rvs(size=size))
            else:
                edges = self.graph.edges(node)
                return [random.choice(edges) for _ in xrange(size)]

        walks = [[] for i in xrange(self.graph.num_nodes)]
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

                if utils.random_true(RESTART_PROB):
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
        sys.stdout.write(".")  # Just to give an indicator of progress

        return float(num_hits) / num_iters


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
        return [self._hitting_time_single(i, pretrust_set, weighted)
                for i in xrange(self.graph.num_nodes)]

    def max_flow(self):
        """ All-pairs maximum flow.

        Uses Ford-Fulkerson with the Edmonds-Karp-Dinitz path selection rule to
        guarantee a running time of O(VE^2).

        Returns:
            An array of n arrays of length n, representing the maximum flow
            that can be pushed to each of the n nodes. None is used for pairs
            where 0 flow can be pushed.
        """
        max_flow_scores = []
        for i in xrange(self.graph.num_nodes):
            neighbor_scores = []
            for j in xrange(self.graph.num_nodes):
                if i == j:
                    neighbor_scores.append(None)
                else:
                    mf = nx.max_flow(self.graph, i, j, capacity='weight')
                    neighbor_scores.append(None if mf == 0 else mf)
            max_flow_scores.append(neighbor_scores)
        return max_flow_scores

    def shortest_path(self):
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
        nx_dict = nx.all_pairs_dijkstra_path_length(
            self.graph, weight='inv_weight')
        # Convert from dict format to array format
        shortest_paths = []
        for i, d in nx_dict.iteritems():
            neighbors = []
            for j in xrange(self.graph.num_nodes):
                try:
                    neighbors.append(1.0 / d[j])
                except (KeyError, ZeroDivisionError):
                    neighbors.append(None)
            shortest_paths.append(neighbors)
        return shortest_paths

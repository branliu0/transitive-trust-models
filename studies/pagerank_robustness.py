import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from trust_graph import TrustGraph

NUM_NODES = 50
NUM_EDGES = 20
NUM_ITERS = 100

def pagerank_outedge_robustness():
    """ Examine how much PageRank varies as we remove outedges for a node. """
    g = TrustGraph(NUM_NODES, 'beta', 'cluster', NUM_EDGES, 'sample', 50)
    node = 0  # Arbitrarily picked, WLOG
    edges = g.out_edges(node, data=True)

    raw_prs = [[] for _ in xrange(NUM_EDGES + 1)]
    for e in xrange(NUM_EDGES + 1):
        sys.stdout.write('.')
        for _ in xrange(NUM_ITERS):
            g.remove_edges_from(g.out_edges(node))  # Remove all outedges
            g.add_edges_from(random.sample(edges, e))  # Add in random subset
            raw_prs[e].append(nx.pagerank_numpy(g)[node])
    prs = np.mean(np.array(raw_prs, dtype=float), axis=1)

    plt.plot(prs)
    plt.suptitle('PageRank robustness: PageRank score as outedges increases')
    plt.xlabel('Number of outedges for node 0')
    plt.ylabel('PageRank score for node 0')


import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from trust_graph import TrustGraph
from trust_models import TrustModels

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


def pagerank_vs_hitting_time(num_graphs):
    graphs = []
    prs = []
    hts = []
    corrs = []
    for i in xrange(num_graphs):
        g = nx.gnp_random_graph(NUM_NODES, float(NUM_EDGES) / NUM_NODES,
                                directed=True)
        graphs.append(g)
        for e in g.edges_iter():
            g[e[0]][e[1]]['weight'] = random.random()

        m = TrustModels(g)

        prs.append(m.pagerank())
        hts.append(m.hitting_pagerank('all'))
        corrs.append(stats.spearmanr(prs[i], hts[i])[0])

        sys.stdout.write('.')

    # Plot correlations
    plt.hist(corrs)

    plt.suptitle('Correlation of PageRank and Eigen Hitting Time\n'
                 'Trials on Erdos-Renyi graphs with %d nodes and prob %0.2f'
                 % (NUM_NODES, float(NUM_EDGES) / NUM_NODES))
    plt.xlabel('Spearman rank-correlation')
    plt.ylabel('Number of graphs (independent trials)')
    plt.margins(0.07)
    plt.show()

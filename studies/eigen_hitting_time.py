import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from trust_models import TrustModels

MC_ITERS = int(1e5)
NUM_NODES = 5
EDGE_PROB = 0.5

def compare_hitting_times(num_graphs):
    graphs = []
    mc_ht = []
    eigen_ht = []
    for _ in xrange(num_graphs):
        g = nx.gnp_random_graph(NUM_NODES, EDGE_PROB, directed=True)
        graphs.append(g)
        # Add random weights
        for e in g.edges_iter():
            g[e[0]][e[1]]['weight'] = random.random()

        m = TrustModels(g)

        mc_ht.append(m.hitting_time('all', num_iters=MC_ITERS))
        eigen_ht.append(m.hitting_pagerank('all'))

    mc_ht = np.transpose(mc_ht)
    eigen_ht = np.transpose(eigen_ht)

    # Plot the actual hitting time scores
    xs = range(1, num_graphs + 1)
    for i in xrange(NUM_NODES):
        plt.plot(xs, mc_ht[i], 'r:')
        plt.plot(xs, eigen_ht[i], 'b:')

    plt.suptitle('Eigen Hitting Time vs. Monte Carlo Hitting Time (%d iters)\n'
                 'Graphs are Erdos-Renyi graphs with %d nodes and %0.2f prob'
                 % (MC_ITERS, NUM_NODES, EDGE_PROB))
    plt.xlabel('Graph number (independent trials)')
    plt.xticks(xs)
    plt.ylabel('Hitting time score')
    plt.margins(0.07)
    plt.show()

    # Plot the residues
    residues = mc_ht - eigen_ht
    for i in xrange(NUM_NODES):
        plt.plot(xs, residues[i])

    plt.suptitle('Residues (Monte Carlo - Eigen)')
    plt.xlabel('Graph number (independent trials)')
    plt.xticks(xs)
    plt.ylabel('Difference in hitting time (Monte Carlo - Eigen)')
    plt.margins(0.07)
    plt.show()

    return graphs

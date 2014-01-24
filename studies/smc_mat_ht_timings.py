import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import hitting_time.mat_hitting_time as m
import hitting_time.single_monte_carlo as smc
from trust_graph import TrustGraph

NODE_COUNTS = [20, 40, 70, 110, 160, 230, 300]


def time_experiment(num_iters, smc_walks=5):
    graphs = [[TrustGraph(num_nodes, 'uniform', 'uniform', num_nodes / 2,
                          'sample', 32) for _ in xrange(num_iters)]
              for num_nodes in NODE_COUNTS]
    mat_times = np.zeros((len(NODE_COUNTS), num_iters))
    smc_times = np.zeros((len(NODE_COUNTS), num_iters))
    corrs = np.zeros((len(NODE_COUNTS), num_iters))

    for i, graph_set in enumerate(graphs):
        for j, g in enumerate(graph_set):
            start_time = time.clock()
            ht_mat = m.personalized_LA_ht(g)
            mat_times[i, j] = time.clock() - start_time
            print '[%d]: mat took %.2f secs' % (len(g), mat_times[i, j])

            start_time = time.clock()
            ht_smc = smc.complete_path_smc_hitting_time(
                g, num_walks=(len(g) * smc_walks))
            smc_times[i, j] = time.clock() - start_time
            print '[%d]: SMC took %.2f secs' % (len(g), smc_times[i, j])

            # Average of all the row correlations
            corrs[i, j] = np.mean(
                [stats.spearmanr(ht_mat[k, :], ht_smc[k, :])[0]
                 for k in xrange(len(ht_smc))])

    avg_mat_times = np.mean(mat_times, axis=1)
    avg_smc_times = np.mean(smc_times, axis=1)
    avg_corrs = np.mean(corrs, axis=1)

    # Plot Timings
    plt.plot(NODE_COUNTS, avg_mat_times, '--^', label='Matrix Algebra')
    plt.plot(NODE_COUNTS, avg_smc_times, '--^', label='Monte Carlo')
    plt.xticks(NODE_COUNTS, NODE_COUNTS)

    plt.suptitle('Timings of Matrix methods vs. Monte Carlo Methods '
                 '(%d MC walks, %d iters)' % (smc_walks, num_iters))
    plt.xlabel('Number of nodes')
    plt.ylabel('Time (sec)')
    plt.legend(loc='best')
    plt.margins(0.07)
    plt.show()

    # Plot Correlations
    plt.plot(NODE_COUNTS, avg_corrs, '--^')
    plt.xticks(NODE_COUNTS, NODE_COUNTS)

    plt.suptitle('Spearman Correlation between Matrix and Monte Carlo Methods '
                 '(%d MC walks, %d iters)' % (smc_walks, num_iters))
    plt.xlabel('Number of nodes')
    plt.ylabel('Spearman Correlation')
    plt.margins(0.07)
    plt.show()


""" Plots the variance of personalized max flow against graph density.

Dec 16, 2013
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiment_sets import EdgeCountExperimentSet
import studies.efficiency as e
from trust_graph import TrustGraph
import trust_models as tm

NUM_NODES = 50


def max_flow_variance(num_iters=5, edge_counts=None):
    if not edge_counts:
        edge_counts = EdgeCountExperimentSet.DEFAULT_EDGE_COUNTS

    raw_scores = np.zeros((len(edge_counts), num_iters, NUM_NODES, NUM_NODES))
    means = np.zeros((len(edge_counts), num_iters))
    # variances = np.zeros((len(edge_counts), num_iters))
    for i, count in enumerate(edge_counts):
        for j in xrange(num_iters):
            g = TrustGraph(NUM_NODES, 'uniform', 'uniform', count, 'noisy', 30)
            at = g.agent_types
            scores = tm.max_flow(g)
            raw_scores[i, j] = scores
            # corrs = [stats.spearmanr(at, score)[0] for score in scores]
            # means[i, j] = np.mean(corrs)
            means[i, j] = e.compute_informativeness(at, scores, False)
            # variances[i, j] = np.var(corrs)
            # variances[i, j] = np.mean(
                # [np.var([x for x in score if x is not None])
                 # for score in scores])

            # for s in scores:
                # plt.plot(count, np.var([x for x in score if x is not None]),
                         # 'o', alpha=0.2)

    # plt.suptitle('Variances of personalized max flow scores '
                 # 'against graph density (n = %d)' % num_iters)
    # plt.ylabel('Variance of max flow scores')
    # plt.xlabel('Number of edges per node')
    # plt.xticks(edge_counts, edge_counts)
    # plt.margins(0.07)
    # plt.show()

    means = means.mean(axis=1)
    # variances = variances.mean(axis=1)

    # Plot Means
    plt.plot(edge_counts, means, 'o--')
    plt.suptitle('Mean Spearman correlation of personalized max flow '
                 'against graph density (n = %d)' % num_iters)
    plt.ylabel('Average Spearman correlation')
    plt.xlabel('Number of edges per node')
    plt.xticks(edge_counts, edge_counts)
    plt.margins(0.07)
    plt.show()

    # Plot Variances
    # plt.plot(edge_counts, variances, 'o--')
    # plt.suptitle('Variance of personalized max flow scores '
                 # 'against graph density (n = %d)' % num_iters)
    # plt.ylabel('Average variance of personalized max flow scores')
    # plt.xlabel('Number of edges per node')
    # plt.xticks(edge_counts, edge_counts)
    # plt.margins(0.07)
    # plt.show()

    return raw_scores

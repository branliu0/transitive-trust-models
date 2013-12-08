import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from hitting_time.eigen_hitting_time import personalized_eigen_ht
from hitting_time.single_monte_carlo import complete_path_smc_hitting_time
from hitting_time.single_monte_carlo import generative_smc_hitting_time
from hitting_time.single_monte_carlo import naive_smc_hitting_time
import utils

NUM_NODES = 50
EDGE_PROB = 0.5

METHODS = [
    # label name, function
    ("Naive", naive_smc_hitting_time),
    ("Complete Path", complete_path_smc_hitting_time),
    ("Generative", generative_smc_hitting_time)
]


def convergence_by_walks(num_iterations, trials=None, filename=None):
    if not trials:
        trials = [1, 2, 3, 5]
    walks = np.array(trials) * NUM_NODES * NUM_NODES

    graphs = [[utils.random_weighted_graph(NUM_NODES, EDGE_PROB)
               for _ in xrange(num_iterations)]
              for _ in trials]

    # Use eigen hitting time to compute a very accurate estimate
    best_estimates = [[personalized_eigen_ht(g) for g in trial]
                      for trial in graphs]

    # Run the simulations
    means = np.zeros((len(METHODS), len(trials)))
    times = np.zeros((len(METHODS), len(trials), num_iterations))
    for ii, (name, func) in enumerate(METHODS):
        print name
        raws = np.zeros((len(trials), num_iterations, NUM_NODES, NUM_NODES))
        for i, t in enumerate(trials):
            for j in xrange(num_iterations):
                start_time = time.clock()
                raws[i][j] = func(graphs[i][j], t)
                times[ii][i][j] = time.clock() - start_time
                sys.stdout.write('.'); sys.stdout.flush()
        sys.stdout.write('\n')

        # Compute errors
        for i, (trial, best) in enumerate(zip(raws, best_estimates)):
            diffs = np.sum(np.absolute(trial - best), axis=(1, 2))
            diffs /= NUM_NODES * NUM_NODES  # avg error per node
            means[ii][i] = np.mean(diffs)

        # Plot the results
        plt.plot(walks, means[ii], 's-', label=name)

    # Additional plotting parameters
    plt.suptitle('Convergence of Monte Carlo Estimators for '
                 'Hitting Time by Number of Random Walks')
    plt.xlabel('Number of random walks (\'000)')
    plt.xticks(walks, map(str, walks / 1000))
    plt.ylabel("Average error from 'best' (%d trials)" % num_iterations)

    plt.legend(loc='best')
    plt.margins(0.07)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

    # Plot runtimes
    avg_times = np.mean(times, axis=2)
    for ii, (name, _) in enumerate(METHODS):
        plt.plot(walks, avg_times[ii], 's-', label=name)
    plt.suptitle('Running Times of Monte Carlo Estimators '
                 'for Hitting Time by Number of Random Walks')
    plt.xlabel('Number of random walks (\'000)')
    plt.xticks(walks, map(str, walks / 1000))
    plt.ylabel('Average runtime for one iteration (sec) (over %d trials)'
               % num_iterations)
    plt.legend(loc='best')
    plt.margins(0.07)

    if filename:
        plt.savefig('time_' + filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

    return means, avg_times


def convergence_by_nodes(num_iterations, nodes=None, num_walks=25000, filename=None):
    if not nodes:
        nodes = [10, 20, 30, 50, 80]

    graphs = [[utils.random_weighted_graph(num_nodes, EDGE_PROB)
               for _ in xrange(num_iterations)]
              for num_nodes in nodes]
    best_estimates = [[personalized_eigen_ht(g) for g in trial]
                      for trial in graphs]

    means = np.zeros((len(METHODS), len(nodes)))
    times = np.zeros((len(METHODS), len(nodes), num_iterations))
    for ii, (name, func) in enumerate(METHODS):
        print name
        for i, n in enumerate(nodes):
            raws = np.zeros((num_iterations, n, n))
            for j in xrange(num_iterations):
                start_time = time.clock()
                raws[j] = func(graphs[i][j], int(num_walks / float(n * n)))
                times[ii][i][j] = time.clock() - start_time
                sys.stdout.write('.'); sys.stdout.flush()
            diffs = np.sum(np.absolute(raws - best_estimates[i]), axis=(1, 2))
            diffs /= n * n
            means[ii][i] = np.mean(diffs)
        sys.stdout.write('\n')

        plt.plot(nodes, means[ii], 's-', label=name)

    plt.suptitle('Error of Monte Carlo Estimators by Graph Size')
    plt.xlabel('Number of nodes')
    plt.xticks(nodes, nodes)
    plt.ylabel("Average error from 'best' (%d trials)" % num_iterations)
    plt.legend(loc='best')
    plt.margins(0.07)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

    # Plot runtimes
    avg_times = np.mean(times, axis=2)
    for ii, (name, _) in enumerate(METHODS):
        plt.plot(nodes, avg_times[ii], 's-', label=name)
    plt.suptitle('Running Times of Monte Carlo Estimators by Graph Size')
    plt.xlabel('Number of nodes')
    plt.xticks(nodes, nodes)
    plt.ylabel('Average runtime for one iteration (sec) (over %d trials)'
               % num_iterations)
    plt.legend(loc='best')
    plt.margins(0.07)

    if filename:
        plt.savefig('time_' + filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

    return means, avg_times

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hitting_time.eigen_hitting_time import personalized_eigen_ht
from hitting_time.single_monte_carlo import complete_path_smc_hitting_time
from hitting_time.single_monte_carlo import generative_smc_hitting_time
from hitting_time.single_monte_carlo import naive_smc_hitting_time
import utils

NUM_NODES = 50
EDGE_PROB = 0.5


def plot_convergence(num_iterations, trials=None, filename=None):
    if not trials:
        trials = [1, 2, 3, 5]
    walks = np.array(trials) * NUM_NODES * NUM_NODES

    graphs = [[utils.random_weighted_graph(NUM_NODES, EDGE_PROB)
               for _ in xrange(num_iterations)]
              for _ in trials]

    # Use eigen hitting time to compute a very accurate estimate
    best_estimates = [[personalized_eigen_ht(g) for g in trial]
                      for trial in graphs]

    methods = [
        # label name, function
        ("Naive", naive_smc_hitting_time),
        ("Complete Path", complete_path_smc_hitting_time),
        ("Generative", generative_smc_hitting_time)
    ]

    # Run the simulations
    for name, func in methods:
        print name
        raws = np.zeros((len(trials), num_iterations, NUM_NODES, NUM_NODES))
        for i, t in enumerate(trials):
            for j in xrange(num_iterations):
                raws[i][j] = func(graphs[i][j], t)
                sys.stdout.write('.'); sys.stdout.flush()
        sys.stdout.write('\n')

        # Compute errors
        best = np.mean(raws[-1], axis=0)
        means = []
        errs = []
        for trial, best in zip(raws, best_estimates):
            diffs = np.sum(np.absolute(trial - best), axis=(1, 2))
            diffs /= NUM_NODES * NUM_NODES  # avg error per node
            means.append(np.mean(diffs))
            errs.append(1.96 * stats.sem(diffs))

        # Plot the results
        plt.plot(walks, means, 's-', label=name)

    # Additional plotting parameters
    plt.suptitle('Convergence of Single-Threaded Monte Carlo Estimators of '
                 'Hitting Time')
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


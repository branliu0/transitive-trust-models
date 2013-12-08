import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    log_walks = np.log10(walks)

    graphs = [[utils.random_weighted_graph(NUM_NODES, EDGE_PROB)
               for _ in xrange(num_iterations)]
              for _ in trials]

    methods = [
        # label name, function
        ("Naive", naive_smc_hitting_time),
        ("Complete Path", complete_path_smc_hitting_time),
        ("Generative", generative_smc_hitting_time)
    ]

    # Run the simulations
    for name, func in methods:
        print name
        raws = []
        for i, t in enumerate(trials):
            trial = []
            for j in xrange(num_iterations):
                trial.append(func(graphs[i][j], t))
                sys.stdout.write('.')
            raws.append(trial)
        sys.stdout.write('\n')

        # Compute errors
        best = np.mean(raws[-1], axis=0)
        means = []
        errs = []
        for trial in raws:
            diffs = np.sum(np.absolute(trial - best), axis=(1, 2))
            means.append(np.mean(diffs))
            errs.append(1.96 * stats.sem(diffs))

        # Plot the results
        plt.plot(log_walks, means, 's-', label=name)

    # Additional plotting parameters
    plt.suptitle('Convergence of Single-Threaded Monte Carlo Estimators of '
                 'Hitting Time')
    plt.xlabel('Number of random walks')
    plt.xticks(log_walks, map(str, walks))
    plt.ylabel('Average error (%d trials)' % num_iterations)

    plt.legend(loc='best')
    plt.margins(0.07)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


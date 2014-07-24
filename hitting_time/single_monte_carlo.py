from collections import defaultdict

import numpy as np

from utils import RandomWalk, random_round


def naive_smc_hitting_time(graph, num_walks, alpha=0.15):
    """
    Naive: Tries `num_trial` random walks for each pair of nodes (i, j).
    SMC: Single-threaded Monte Carlo
    """
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))

    walks_per_pair = num_walks / float(N * N)

    for i in xrange(N):
        for j in xrange(N):
            # If we start here, we're already here!
            if i == j:
                hitting_time[i][j] = 1
                continue

            nwalks = random_round(walks_per_pair)
            for _ in xrange(nwalks):
                node = i
                while True:
                    if node == j:
                        hitting_time[i][j] += 1
                        break

                    if walk.terminates() or node is None:
                        break

                    node = walk.step(node)
            hitting_time[i][j] /= nwalks

    return hitting_time


def multihit_smc_hitting_time(graph, num_walks, alpha=0.15):
    # ALERT: This currently doesn't return the correct value for dangling nodes
    # Look into this later when it becomes important.
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))

    for i in xrange(N):
        nwalks = random_round(float(num_walks) / N)
        for _ in xrange(nwalks):
            node = i
            hits = np.zeros(N)
            while True:
                hits[node] = 1
                if walk.terminates() or node is None:
                    break
                node = walk.step(node)
            hitting_time[i] += hits
        hitting_time[i] /= nwalks

    return hitting_time


def multiwalk_smc_hitting_time(graph, num_walks, alpha=0.15):
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hits = np.zeros((N, N))
    walks = np.zeros(N)

    for i in xrange(N):
        for _ in xrange(random_round(float(num_walks) / N)):
            # First simulate the random walk
            node = i
            steps = []
            counter = defaultdict(int)
            while True:
                steps.append(node)
                counter[node] += 1
                node = walk.step(node)
                if walk.terminates() or node is None:
                    break
            # Extract the analysis from the walk afterward, all at once
            seen = np.repeat(False, N)
            for n in reversed(steps):
                if not seen[n]:
                    seen[n] = True
                    hits[counter.keys(), n] += counter.values()
                    walks[n] += counter[n]
                counter[n] -= 1

    return hits / np.outer(walks, np.ones(N))

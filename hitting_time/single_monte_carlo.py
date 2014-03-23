from collections import defaultdict

import numpy as np

from utils import RandomWalk, random_round


def naive_smc_hitting_time(graph, num_walks=None, num_trials=None, alpha=0.15):
    """
    Naive: Tries `num_trial` random walks for each pair of nodes (i, j).
    SMC: Single-threaded Monte Carlo
    """
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))

    if num_trials is None:
        num_trials = num_walks / float(N * N)

    for i in xrange(N):
        for j in xrange(N):
            # If we start here, we're already here!
            if i == j:
                hitting_time[i][j] = 1
                continue

            for _ in xrange(random_round(num_trials)):
                node = i
                while True:
                    if node == j:
                        hitting_time[i][j] += 1
                        break

                    if walk.terminates():
                        break

                    node = walk.step(node)
            hitting_time[i][j] /= num_trials

    return hitting_time


def naive_psmc_hitting_time(graph, num_trials, alpha=0.15):
    """
    Naive: Tries `num_trial` random walks for each pair of nodes (i, j).
    PSMC: Parallel Single-threaded Monte Carlo. Tries to do this with more
    memory locality.
    """
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))
    tokens = [[] for _ in xrange(N)]

    # Initialize tokens
    for i in xrange(N):
        for j in xrange(N):
            tokens[i].extend([(i, j) for _ in xrange(num_trials)])

    # Iterate until done
    more_moves = True
    while more_moves:
        more_moves = False
        for i in xrange(N):
            while tokens[i]:
                t = tokens[i].pop()
                if t[1] == i:
                    hitting_time[t[0]][t[1]] += 1
                    continue

                if walk.terminates():
                    continue

                tokens[walk.step(i)].append(t)
                more_moves = True

    hitting_time /= num_trials
    return hitting_time


def multihit_smc_hitting_time(graph, num_trials=None, num_walks=None,
                                   alpha=0.15):
    if num_trials is None and num_walks is None:
        raise ValueError("Must specify one of num_trials or num_walks")

    N = graph.number_of_nodes()
    if num_walks is None:
        num_walks = N * N * num_trials
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))

    for i in xrange(N):
        for _ in xrange(random_round(float(num_walks) / N)):
            node = i
            hits = np.zeros(N)
            while True:
                hits[node] = 1
                if walk.terminates():
                    break
                node = walk.step(node)
            hitting_time[i] += hits

    hitting_time /= (num_walks / N)
    return hitting_time


def multiwalk_smc_hitting_time(graph, num_trials=None, num_walks=None,
                                alpha=0.15):
    if num_trials is None and num_walks is None:
        raise ValueError("Must specify one of num_trials or num_walks")

    N = graph.number_of_nodes()
    if num_walks is None:
        num_walks = N * N * num_trials
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
                if walk.terminates():
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

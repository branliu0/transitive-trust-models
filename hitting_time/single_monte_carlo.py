import numpy as np

from utils import RandomWalk


def naive_smc_hitting_time(graph, num_trials, alpha=0.15):
    """
    Naive: Tries `num_trial` random walks for each pair of nodes (i, j).
    SMC: Single-threaded Monte Carlo
    """
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))

    for i in xrange(N):
        for j in xrange(N):
            # If we start here, we're already here!
            if i == j:
                hitting_time[i][j] = 1
                continue

            for _ in xrange(num_trials):
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

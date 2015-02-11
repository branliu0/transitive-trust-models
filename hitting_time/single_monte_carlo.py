from collections import defaultdict
import time

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


def algo4(graph, num_walks, alpha=0.15):
    N = graph.number_of_nodes()
    walk = RandomWalk(graph, alpha)
    hits = np.zeros((N, N))
    walks = np.zeros(N)

    walk_hits = np.zeros((N, N))
    walk_visits = np.zeros(N)

    for i in xrange(N):
        for _ in xrange(random_round(float(num_walks) / N)):
            walk_hits[:] = 0
            walk_visits[:] = 0
            steps = []
            node = i
            while True:
                steps.append(node)
                node = walk.step(node)
                if walk.terminates() or node is None:
                    break

            for ii, v_i in enumerate(steps):
                walk_visits[v_i] = 1
                for jj, v_j in enumerate(steps[ii:]):
                    walk_hits[v_i, v_j] = 1

            hits += walk_hits
            walks += walk_visits

    return np.divide(hits, np.outer(walks, np.ones(N)), dtype=float)

###############################################################################
###############################################################################
###############################################################################


def timed_naive_smc_hitting_time(graph, time_limit, alpha=0.15):
    """
    Naive: Tries `num_trial` random walks for each pair of nodes (i, j).
    SMC: Single-threaded Monte Carlo
    """
    N = graph.number_of_nodes()
    walk_count = 0
    done = False
    start_time = time.clock()
    walk = RandomWalk(graph, alpha)
    hits = np.zeros((N, N))
    walks = np.zeros((N, N))

    while not done:
        for i in xrange(N):
            if done: break
            for j in xrange(N):
                walks[i, j] += 1
                walk_count += 1
                # If we start here, we're already here!
                if i == j:
                    hits[i, j] += 1
                    continue

                node = i
                while True:
                    if node == j:
                        hits[i, j] += 1
                        break
                    if walk.terminates() or node is None:
                        break
                    node = walk.step(node)

                elapsed = time.clock() - start_time
                if elapsed > time_limit:
                    print 'Naive: Simulated %d walks' % walk_count
                    done = True
                    break

        return np.divide(hits, walks, dtype=float)


def timed_multihit_smc_hitting_time(graph, time_limit, alpha=0.15):
    # ALERT: This currently doesn't return the correct value for dangling nodes
    # Look into this later when it becomes important.
    N = graph.number_of_nodes()
    walk_count = 0
    done = False
    start_time = time.clock()
    walk = RandomWalk(graph, alpha)
    hitting_time = np.zeros((N, N))
    walks = np.zeros(N)

    walk_hits = np.zeros(N)

    while not done:
        for i in xrange(N):
            node = i
            walks[i] += 1
            walk_hits[:] = 0
            while True:
                walk_hits[node] = 1
                node = walk.step(node)
                if walk.terminates() or node is None:
                    break
            hitting_time[i] += walk_hits

            walk_count += 1
            elapsed = time.clock() - start_time
            if elapsed > time_limit:
                print 'Multihit: Simulated %d walks' % walk_count
                done = True
                break

    return np.divide(hitting_time, walks, dtype=float)


def timed_multiwalk_smc_hitting_time(graph, time_limit=0, alpha=0.15):
    N = graph.number_of_nodes()
    walk_count = 0
    done = False
    start_time = time.clock()
    walk = RandomWalk(graph, alpha)
    hits = np.zeros((N, N))
    walks = np.zeros(N)

    seen = np.repeat(False, N)

    while not done:
        for i in xrange(N):
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
            seen[:] = False
            for n in reversed(steps):
                if not seen[n]:
                    seen[n] = True
                    hits[counter.keys(), n] += counter.values()
                    walks[n] += counter[n]
                counter[n] -= 1

            walk_count += 1
            elapsed = time.clock() - start_time
            if elapsed > time_limit:
                print 'Multiwalk: Simulated %d walks' % walk_count
                done = True
                break

    return hits / np.outer(walks, np.ones(N))


def timed_algo4(graph, time_limit, alpha=0.15):
    N = graph.number_of_nodes()
    walk_count = 0
    done = False
    start_time = time.clock()
    walk = RandomWalk(graph, alpha)
    hits = np.zeros((N, N))
    walks = np.zeros(N)

    walk_hits = np.zeros((N, N))
    walk_visited = np.zeros(N)

    while not done:
        for i in xrange(N):
            walk_hits[:] = 0
            walk_visited[:] = 0
            steps = []
            node = i
            while True:
                steps.append(node)
                node = walk.step(node)
                if walk.terminates() or node is None:
                    break

            for ii, v_i in enumerate(steps):
                walk_visited[v_i] = 1
                for jj, v_j in enumerate(steps[ii:]):
                    walk_hits[v_i, v_j] = 1

            hits += walk_hits
            walks += walk_visited

            walk_count += 1
            elapsed = time.clock() - start_time
            if elapsed > time_limit:
                print 'Algo4 (new): Simulated %d walks' % walk_count
                done = True
                break

    return np.divide(hits, np.outer(walks, np.ones(N)), dtype=float)

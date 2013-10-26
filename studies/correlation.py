"""
Studying the difference between Kendall-tau rank correlation and Pearson
correlation.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N = 20

def _plot(pearsons, kendalls, title, xlabel):
    plt.plot(pearsons, 'r-s', label='Pearson')
    plt.plot(kendalls, 'b-^', label='Kendall tau')

    plt.xlabel(xlabel)
    plt.ylabel('Correlation (Pearson & Kendall tau)')
    plt.legend(loc='best')
    plt.margins(0.05)
    plt.suptitle(title)

    plt.show()

def high_vs_low_swaps():
    """ Make adjacent swaps at various indices and compare.

    (1) Both Pearson and KT are invariant to the index of the swap
    (2) KT is lower than Pearson for a single swap.
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N - 1)]
    for i, xs in enumerate(swapped):
        xs[i], xs[i + 1] = xs[i + 1], xs[i]
    pearsons = [stats.pearsonr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(pearsons, kendalls,
          ('Pearson vs. KT: Differences in adjacent swaps at different indices\n'
           'SWAP(a[i], a[i+1])'),
          'Index of pair swap')
    # return pearsons, kendalls


def swaps_with_first():
    """ Make swaps with the first element

    (1) As the size of the swap increases, pearson falls quadratically, whereas
    Kendall tau falls linearly
    (2) For N = 20, there is a crossover between 12 and 13, before which Pearson
    has a higher correlation, and after which Pearson has a lower correlation
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N - 1)]
    for i, xs in enumerate(swapped):
        xs[0], xs[i + 1] = xs[i + 1], xs[0]
    pearsons = [stats.pearsonr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(pearsons, kendalls,
          ('Pearson vs. KT: Swapping with first element (varying size of swap)\n'
           'SWAP(a[0], a[i + 1])'),
          'Index of swapped element (with first element)')


def swaps_from_center():
    """ Make progressively larger swaps emanating from the center

    In general, we see a similar result as swaps_with_first -- Kentall tau
    increases linearly while Pearson increases more quadratically.
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N / 2)]
    for i, xs in enumerate(swapped):
        xs[i], xs[N - i - 1] = xs[N - i - 1], xs[i]
    pearsons = [stats.pearsonr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(pearsons, kendalls,
          ('Pearson vs. KT: Swaps emanating from center\n'
           'SWAP(a[i], a[N - i - 1])'),
          'Index of first swapped element')


def random_swaps():
    """ Make an increasing number of random swaps

    Not too much can be discerned, except pearson has a consistently higher
    score than kendall tau.
    """
    TRIALS = 100
    truth = range(N)
    pearsons = []; kendalls = []
    for num_swaps in xrange(N):
        tmp_pearsons = []; tmp_kendalls = []
        for _ in xrange(TRIALS):
            ranks = range(N)
            for _ in xrange(num_swaps):
                a, b = random.randrange(N), random.randrange(N)
                ranks[a], ranks[b] = ranks[b], ranks[a]
            tmp_pearsons.append(stats.pearsonr(truth, ranks)[0])
            tmp_kendalls.append(stats.kendalltau(truth, ranks)[0])
        pearsons.append(np.mean(tmp_pearsons))
        kendalls.append(np.mean(tmp_kendalls))

    _plot(pearsons, kendalls,
          'Pearson vs. KT: Random number of swaps (n = %d)' % TRIALS,
          'Number of random swaps')


def adjacent_swaps():
    """ Make increasingly non-overlapping adjacent swaps

    (1) Both decrease linearly
    (2) Kendall tau decreases more sharply than Pearson
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange((N / 2) + 1)]
    for i, xs in enumerate(swapped):
        for j in xrange(i):
            xs[2 * j], xs[2 * j + 1] = xs[2 * j + 1], xs[2 * j]

    pearsons = [stats.pearsonr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(pearsons, kendalls,
          ('Pearson vs. KT: Non-overlapping adjacent swaps\n'
           'SWAP(a[2 * i], a[2 * i + 1])'),
          'Number of adjacent swaps')

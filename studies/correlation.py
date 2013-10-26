"""
Studying the difference between Kendall-tau rank correlation and Spearman
correlation.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N = 20

def _plot(spearmans, kendalls, title, xlabel):
    plt.plot(spearmans, 'r-s', label='Spearman')
    plt.plot(kendalls, 'b-^', label='Kendall tau')

    plt.xlabel(xlabel)
    plt.ylabel('Correlation (Spearman & Kendall tau)')
    plt.legend(loc='best')
    plt.margins(0.05)
    plt.suptitle(title)

    plt.show()

def high_vs_low_swaps():
    """ Make adjacent swaps at various indices and compare.

    (1) Both Spearman and KT are invariant to the index of the swap
    (2) KT is lower than Spearman for a single swap.
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N - 1)]
    for i, xs in enumerate(swapped):
        xs[i], xs[i + 1] = xs[i + 1], xs[i]
    spearmans = [stats.spearmanr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(spearmans, kendalls,
          ('Spearman vs. KT: Differences in adjacent swaps at different indices\n'
           'SWAP(a[i], a[i+1])'),
          'Index of pair swap')
    # return spearmans, kendalls


def swaps_with_first():
    """ Make swaps with the first element

    (1) As the size of the swap increases, spearman falls quadratically, whereas
    Kendall tau falls linearly
    (2) For N = 20, there is a crossover between 12 and 13, before which Spearman
    has a higher correlation, and after which Spearman has a lower correlation
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N - 1)]
    for i, xs in enumerate(swapped):
        xs[0], xs[i + 1] = xs[i + 1], xs[0]
    spearmans = [stats.spearmanr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(spearmans, kendalls,
          ('Spearman vs. KT: Swapping with first element (varying size of swap)\n'
           'SWAP(a[0], a[i + 1])'),
          'Index of swapped element (with first element)')


def swaps_from_center():
    """ Make progressively larger swaps emanating from the center

    In general, we see a similar result as swaps_with_first -- Kentall tau
    increases linearly while Spearman increases more quadratically.
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange(N / 2)]
    for i, xs in enumerate(swapped):
        xs[i], xs[N - i - 1] = xs[N - i - 1], xs[i]
    spearmans = [stats.spearmanr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(spearmans, kendalls,
          ('Spearman vs. KT: Swaps emanating from center\n'
           'SWAP(a[i], a[N - i - 1])'),
          'Index of first swapped element')


def random_swaps():
    """ Make an increasing number of random swaps

    Not too much can be discerned, except spearman has a consistently higher
    score than kendall tau.
    """
    TRIALS = 100
    truth = range(N)
    spearmans = []; kendalls = []
    for num_swaps in xrange(N):
        tmp_spearmans = []; tmp_kendalls = []
        for _ in xrange(TRIALS):
            ranks = range(N)
            for _ in xrange(num_swaps):
                a, b = random.randrange(N), random.randrange(N)
                ranks[a], ranks[b] = ranks[b], ranks[a]
            tmp_spearmans.append(stats.spearmanr(truth, ranks)[0])
            tmp_kendalls.append(stats.kendalltau(truth, ranks)[0])
        spearmans.append(np.mean(tmp_spearmans))
        kendalls.append(np.mean(tmp_kendalls))

    _plot(spearmans, kendalls,
          'Spearman vs. KT: Random number of swaps (n = %d)' % TRIALS,
          'Number of random swaps')


def adjacent_swaps():
    """ Make increasingly non-overlapping adjacent swaps

    (1) Both decrease linearly
    (2) Kendall tau decreases more sharply than Spearman
    """
    truth = range(N)
    swapped = [range(N) for _ in xrange((N / 2) + 1)]
    for i, xs in enumerate(swapped):
        for j in xrange(i):
            xs[2 * j], xs[2 * j + 1] = xs[2 * j + 1], xs[2 * j]

    spearmans = [stats.spearmanr(truth, xs)[0] for xs in swapped]
    kendalls = [stats.kendalltau(truth, xs)[0] for xs in swapped]

    _plot(spearmans, kendalls,
          ('Spearman vs. KT: Non-overlapping adjacent swaps\n'
           'SWAP(a[2 * i], a[2 * i + 1])'),
          'Number of adjacent swaps')

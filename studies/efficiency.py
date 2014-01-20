import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from scipy import stats

import strategic_trust_models as stm
from trust_graph import TrustGraph


# Format:
# 1. is_global? (Boolean)
# 2. name (String)
# 3. corresponding function (performs manipulations and computes score) (func)
MECHANISMS = [
    (True, 'global_pagerank', stm.global_pagerank),
    (False, 'person_pagerank', stm.person_pagerank),
    (True, 'global_hitting_time', stm.global_hitting_time),
    (False, 'person_hitting_time', stm.person_hitting_time),
    (False, 'person_max_flow', stm.person_max_flow),
    (False, 'person_shortest_path', stm.person_shortest_path),
]
NAMES = [x[1] for x in MECHANISMS]

PLOT_LABELS = {
    'global_pagerank': '--*',
    'person_pagerank': '--*',
    'global_hitting_time': '--*',
    'person_hitting_time': '--^',
    'person_max_flow': '--^',
    'person_shortest_path': '--s',
}

NUM_NODES = 50
NUM_EDGES = 15
NUM_SAMPLES = 32
SYBIL_PCT = 0.50
DEFAULT_STRATEGIC_COUNTS = [0, 3, 5, 10, 20]
NUM_CHOICES = 5


NUM_NODES = 30
NUM_EDGES = 15
DEFAULT_STRATEGIC_COUNTS = [0, 3, 5, 10, 15, 20, 25]


def compute_informativeness(agent_types, scores, is_global):
    if is_global:
        # Spearman's rho
        corr, _ = stats.spearmanr(agent_types, scores)
        return corr
    else:
        # print scores.round(2)
        # Take N individual Spearman correlations and take the average.
        corrs = np.zeros(len(agent_types))
        for i, row in enumerate(scores):
            # Right now, I'm removing the None values one's one value, but
            # we may consider changing this.
            removed_indices = [j for j, x in enumerate(row) if x is None]
            if removed_indices:
                print 'Warning: saw %d None values' % len(removed_indices)
            removed_indices += [i]
            ats = [val for j, val in enumerate(agent_types) if j not in removed_indices]
            vals = [val for j, val in enumerate(row) if j not in removed_indices]
            corrs[i], _ = stats.spearmanr(ats, vals)
            # print corrs[i], ats, vals
        # print corrs
        return corrs.mean()


def compute_efficiency(agent_types, scores, is_global, K=NUM_CHOICES):
    N = len(agent_types)
    def prob(i):
        # Probability of the i-th index element being the highest ranked element
        # in a random NUM_CHOICES-sample from NUM_NODES elements.
        # i \in 1, 2, ..., NUM_NODES
        return (float(K) / N) * (
            scipy.misc.comb(i, K - 1) / scipy.misc.comb(N - 1, K - 1))

    if is_global:
        scores = np.repeat([scores], N, axis=0)
        # ranks = stats.rankdata(scores, method='max')
        # return sum(prob(ranks[i]) * agent_types[i] for i in xrange(N))

    effs = np.zeros(N)
    for i, row in enumerate(scores):
        removed_indices = [j for j, x in enumerate(row) if x is None]
        if removed_indices:
            print 'Warning: saw %d None values' % len(removed_indices)
        removed_indices += [i]
        vals = [val for j, val in enumerate(row) if j not in removed_indices]
        ats = [val for j, val in enumerate(agent_types) if j not in removed_indices]
        ranks = stats.rankdata(vals, method='max')
        effs[i] = sum(prob(rank) * ats[j]
                        for j, rank in enumerate(ranks))
    return effs.mean()


def efficiency_experiments(num_iters, strategic_counts=None):
    """
    1. Compute scores under manipulations.
    2. Compute informativeness WRT % of strategic agents.
    3. Compute efficiency WRT % of strategic agents.
    """
    if not strategic_counts:
        strategic_counts = DEFAULT_STRATEGIC_COUNTS
    graphs = [[TrustGraph(NUM_NODES, 'uniform', 'uniform', NUM_EDGES,
                          'noisy', NUM_SAMPLES) for _ in xrange(num_iters)]
              for _ in xrange(len(strategic_counts))]
    informativeness = {n: np.zeros(len(strategic_counts)) for n in NAMES}
    efficiency = {n: np.zeros(len(strategic_counts)) for n in NAMES}

    for i, num_strategic in enumerate(strategic_counts):
        for is_global, name, func in MECHANISMS:
            info, eff = np.zeros(num_iters), np.zeros(num_iters)
            for j in xrange(num_iters):
                g = graphs[i][j]
                scores = func(g, num_strategic, SYBIL_PCT)
                info[j] = compute_informativeness(g.agent_types, scores, is_global)
                eff[j] = compute_efficiency(g.agent_types, scores, is_global)
            informativeness[name][i] = info.mean()
            efficiency[name][i] = eff.mean()

    return {'info': informativeness,
            'eff': efficiency,
            'num_iters': num_iters}

def plot(info, eff, num_iters, strategic_counts=None):
    if not strategic_counts:
        strategic_counts = DEFAULT_STRATEGIC_COUNTS

    # Plotting Informativeness
    for n in NAMES:
        plt.plot(strategic_counts, info[n], PLOT_LABELS[n], label=n)

    plt.suptitle('Informativeness of TTMs under manipulations\n'
                 '%d nodes, %d edges/node, %d%% sybils (%d iters)' % (
                    NUM_NODES, NUM_EDGES, int(100 * SYBIL_PCT), num_iters))
    plt.xticks(strategic_counts, strategic_counts)
    plt.xlabel('Number of Strategic Agents')
    plt.ylabel('Informativeness (Speaman\'s rho)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fancybox=True, shadow=True)
    plt.margins(0.07)
    plt.show()

    # Plotting Efficiency
    for n in NAMES:
        plt.plot(strategic_counts, eff[n], PLOT_LABELS[n], label=n)

    plt.suptitle('Efficiency of TTMs under manipulations\n'
                 '%d nodes, %d edges/node, %d%% sybils, K = %d (%d iters)' % (
                    NUM_NODES, NUM_EDGES, int(100 * SYBIL_PCT), NUM_CHOICES, num_iters))
    plt.xticks(strategic_counts, strategic_counts)
    plt.xlabel('Number of Strategic Agents')
    plt.ylabel('Efficiency')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fancybox=True, shadow=True)
    plt.margins(0.07)
    plt.show()


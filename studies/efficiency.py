import time

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
    # (True, 'global_hitting_time', stm.global_smc_hitting_time),
    (False, 'person_hitting_time', stm.person_hitting_time),
    # (False, 'person_hitting_time', stm.person_smc_hitting_time),
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
DEFAULT_STRATEGIC_COUNTS = [0, 1, 2, 3, 4, 5, 10, 20, 25]
DEFAULT_SYBIL_PCTS = [0, 0.03, 0.07, 0.10, 0.13, 0.16, 0.2, 0.4, 0.6]
NUM_CHOICES = 5


NUM_NODES = 30
NUM_EDGES = 15
DEFAULT_STRATEGIC_COUNTS = [0, 3, 5, 10, 15, 20, 25]


def compute_scores(stm_func, graph, num_strategic, sybil_pct):
    return stm_func(graph, num_strategic, sybil_pct)


def compute_informativeness(agent_types, scores, is_global):
    # Add some noise to avoid ties
    scores += np.random.normal(scale=1e-4, size=scores.shape)

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


def compute_efficiency(agent_types, scores, is_global, strategic_agents=[], K=NUM_CHOICES):
    N = len(agent_types)
    def prob(i):
        # Probability of the i-th index element being the highest ranked element
        # in a random NUM_CHOICES-sample from NUM_NODES elements.
        # i \in 0, 1, 2, ..., NUM_NODES - 1
        return (float(K) / N) * (
            scipy.misc.comb(i, K - 1) / scipy.misc.comb(N - 1, K - 1))

    # Add a tiny bit of noise so that we don't get ties in our rankings
    scores += np.random.normal(scale=1e-4, size=scores.shape)

    if is_global:
        scores = np.repeat([scores], N, axis=0)
    effs = []
    for i, row in enumerate(scores):
        # Only average over non-strategic agents
        if i in strategic_agents:
            continue

        removed_indices = [j for j, x in enumerate(row) if x is None]
        if removed_indices:
            print 'Warning: saw %d None values' % len(removed_indices)
        removed_indices += [i]
        vals = [val for j, val in enumerate(row) if j not in removed_indices]
        ats = [val for j, val in enumerate(agent_types) if j not in removed_indices]
        ranks = stats.rankdata(vals, method='max')
        effs.append(sum(prob(rank) * ats[j]
                        for j, rank in enumerate(ranks)))
    return np.mean(effs)


def efficiency_by_sybil_pct(num_iters, num_strategic=None, sybil_pcts=None,
                            cutlinks=True, gensybils=True):
    if not sybil_pcts:
        sybil_pcts = DEFAULT_SYBIL_PCTS
    if num_strategic is None:
        num_strategic = NUM_NODES / 2
    graphs = [[TrustGraph(NUM_NODES, 'uniform', 'uniform', NUM_EDGES,
                          'noisy', NUM_SAMPLES) for _ in xrange(num_iters)]
              for _ in xrange(len(sybil_pcts))]
    informativeness = {n: np.zeros(len(sybil_pcts)) for n in NAMES}
    efficiency = {n: np.zeros(len(sybil_pcts)) for n in NAMES}

    for i, sybil_pct in enumerate(sybil_pcts):
        for is_global, name, func in MECHANISMS:
            info, eff = np.zeros(num_iters), np.zeros(num_iters)
            for j in xrange(num_iters):
                g = graphs[i][j]

                start_time = time.clock()

                scores = func(g, num_strategic, sybil_pct, cutlinks, gensybils)
                info[j] = compute_informativeness(g.agent_types, scores, is_global)
                eff[j] = compute_efficiency(g.agent_types, scores, is_global)

                total_time = time.clock() - start_time
                print '%s took %.2f secs' % (name, total_time)
            informativeness[name][i] = info.mean()
            efficiency[name][i] = eff.mean()

    return {'info': informativeness,
            'eff': efficiency,
            'xticks': sybil_pcts,
            'xlabel': '% Sybils created per strategic agent',
            'subtitle': '%d nodes, %d edges/node, %d strategic, (%d iters)' % (
                    NUM_NODES, NUM_EDGES, num_strategic, num_iters)}


def efficiency_by_strategic_counts(num_iters, strategic_counts=None,
                                   cutlinks=True, gensybils=True):
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
                scores = func(g, num_strategic, SYBIL_PCT, cutlinks, gensybils)
                info[j] = compute_informativeness(g.agent_types, scores, is_global)
                eff[j] = compute_efficiency(g.agent_types, scores, is_global)
            informativeness[name][i] = info.mean()
            efficiency[name][i] = eff.mean()

    return {'info': informativeness,
            'eff': efficiency,
            'xticks': strategic_counts,
            'xlabel': 'Number of Strategic Agents',
            'subtitle': '%d nodes, %d edges/node, %d%% sybils (%d iters)' % (
                    NUM_NODES, NUM_EDGES, int(100 * SYBIL_PCT), num_iters)}


def efficiency_by_edge_count(num_iters, edge_counts, num_strategic, sybil_pct,
                             cutlinks=True, gensybils=True):
    graphs = [[TrustGraph(NUM_NODES, 'uniform', 'uniform', e, 'noisy',
                          NUM_SAMPLES) for _ in xrange(num_iters)]
              for e in edge_counts]
    informativeness = {n: np.zeros(len(edge_counts)) for n in NAMES}
    efficiency = {n: np.zeros(len(edge_counts)) for n in NAMES}

    for i, _ in enumerate(edge_counts):
        for is_global, name, func in MECHANISMS:
            info, eff = np.zeros(num_iters), np.zeros(num_iters)
            for j in xrange(num_iters):
                g = graphs[i][j]
                scores = func(g, num_strategic, sybil_pct, cutlinks, gensybils)
                info[j] = compute_informativeness(g.agent_types, scores, is_global)
                eff[j] = compute_efficiency(g.agent_types, scores, is_global)
            informativeness[name][i] = info.mean()
            efficiency[name][i] = eff.mean()

    return {'info': informativeness,
            'eff': efficiency,
            'xticks': edge_counts,
            'xlabel': 'Number of edges per node',
            'subtitle': '%d nodes, %d strategic, %d%% sybils (%d iters)' % (
                    NUM_NODES, num_strategic, int(100 * sybil_pct), num_iters)}



def plot(info, eff, xticks, xlabel, subtitle):
    # Plotting Informativeness
    for n in NAMES:
        plt.plot(xticks, info[n], PLOT_LABELS[n], label=n)

    plt.suptitle('Informativeness of TTMs under manipulations\n' + subtitle)
    plt.xticks(xticks, xticks)
    plt.xlabel(xlabel)
    plt.ylabel('Informativeness (Speaman\'s rho)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fancybox=True, shadow=True)
    plt.margins(0.07)
    plt.show()

    # Plotting Efficiency
    for n in NAMES:
        plt.plot(xticks, eff[n], PLOT_LABELS[n], label=n)

    plt.suptitle('Efficiency of TTMs under manipulations\n' + subtitle)
    plt.xticks(xticks, xticks)
    plt.xlabel(xlabel)
    plt.ylabel('Efficiency (K = %d)' % NUM_CHOICES)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fancybox=True, shadow=True)
    plt.margins(0.07)
    plt.show()


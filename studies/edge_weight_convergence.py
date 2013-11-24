import math

import matplotlib.pyplot as plt
import numpy as np

from trust_graph import TrustGraph

def edge_weight_convergence(agent_type_prior, edge_weight_strategy,
                            samples=[1, 2, 4, 8, 16, 32, 64, 128],
                            num_draws=1000, agent_types=[0.25, 0.75]):
    if agent_type_prior not in TrustGraph.AGENT_TYPE_PRIORS:
        raise ValueError("Invalid agent type prior")
    if edge_weight_strategy not in TrustGraph.EDGE_WEIGHT_STRATEGIES:
        raise ValueError("Invalid edge weight strategy")

    sample_draws = [
        [TrustGraph._sampled_edge_weight(
            agent_types, edge_weight_strategy, agent_type_prior, s, 0, 1)
         for _ in xrange(num_draws)]
        for s in samples]

    draw_means = [np.mean(draws) for draws in sample_draws]
    draw_error = [1.96 * np.std(draws) for draws in sample_draws]
    expected_weight = TrustGraph._expected_edge_weight(
        agent_types, edge_weight_strategy, agent_type_prior, 0, 1)
    true_weight = agent_types[1]
    log_samples = [math.log(s, 2) for s in samples]

    # Draw the graph
    plt.errorbar(log_samples, draw_means, draw_error, fmt='rs',
                 label='Sampled weights')
    plt.axhline(expected_weight, color='m', linestyle=':',
                label='Expected weight')
    plt.axhline(true_weight, color='b', linestyle='--',
                label='Correct weight')

    plt.suptitle('Convergence of sampled edge weights (%d draws)\n'
                 "'%s' agent prior type, '%s' edge weight strategy" %
                 (num_draws, agent_type_prior, edge_weight_strategy))
    plt.xlabel('log(Number of samples)')
    plt.ylabel('Edge weight (95% Confidence Intervals)')

    plt.xticks(log_samples, samples)
    plt.margins(0.07)
    plt.legend(loc='best')

    plt.show()


def all_convergences():
    for agent_type_prior in TrustGraph.AGENT_TYPE_PRIORS:
        for edge_weight_strategy in TrustGraph.EDGE_WEIGHT_STRATEGIES:
            edge_weight_convergence(agent_type_prior, edge_weight_strategy)

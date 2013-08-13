from collections import defaultdict

import numpy as np
from scipy import stats

from trust_graph import TrustGraph
from trust_models import TrustModels


class Experiment(object):
    """ Represents one interation of a contained experiement with parameters.

    An Experiment in some sense orchestrates the connections between other
    individual components, and is responsible for measurement. An Experiment
    basically does three things:
        1. Generate the graph
        2. Compute the trust model scores
        3. Measure the informativeness of those scores.
    """

    def __init__(self, num_nodes, agent_type_prior, edge_strategy,
                 edges_per_node, edge_weight_strategy, num_weight_samples):
        """
        Args:
            (These are exactly the same as TrustGraph. Please refer there for
            details)
        """
        self.graph = TrustGraph(
            num_nodes, agent_type_prior, edge_strategy,
            edges_per_node, edge_weight_strategy, num_weight_samples)
        self.trust_models = TrustModels(self.graph)

        self.global_ttms = defaultdict(dict)
        self.personalized_ttms = defaultdict(dict)
        self.correlations = {
            'pearson': stats.pearsonr,
            'kendalltau': stats.kendalltau,
            'spearman': stats.spearmanr
        }

    def compute_informativeness(self):
        """ Compute trust model scores and measure informativeness. """
        self.compute_scores()
        self.measure_informativeness()

    def compute_scores(self):
        """ Actually run the trust model routines. Can take a while. """
        self.global_ttms['pagerank_weighted']['scores'] = \
                self.trust_models.pagerank(weighted=True)
        self.global_ttms['hitting_time_weighted_all']['scores'] = \
                self.trust_models.hitting_time('all', weighted=True)
        self.global_ttms['hitting_time_weighted_prob']['scores'] = \
                self.trust_models.hitting_time('prob', weighted=True)
        self.global_ttms['hitting_time_weighted_top']['scores'] = \
                self.trust_models.hitting_time('top', weighted=True)

        self.personalized_ttms['max_flow']['scores'] = \
                self.trust_models.max_flow()
        self.personalized_ttms['shortest_path']['scores'] = \
                self.trust_models.shortest_path()

    def measure_informativeness(self):
        """ Use correlation functions to measure the informativeness of scores.

        This function should be run after compute_scores() is executed. This
        function uses three measures of correlation:
            1. Pearson product-moment correlation coefficient
            2. Kendall tau rank correlation coefficient
            3. Spearman's rank correlation coefficient
        """
        at = self.graph.agent_types
        for model in self.global_ttms.values():
            for corrname, corr in self.correlations.items():
                model[corrname], _ = corr(at, model['scores'])

        for model in self.personalized_ttms.values():
            for corrname, corr in self.correlations.items():
                model[corrname] = {}
                model[corrname]['values'] = []
                for row in model['scores']:
                    # Only correlate values that are not None
                    none_indices = [i for i, x in enumerate(row) if x is None]
                    corrval, _ = corr(
                        [val for i, val in enumerate(at) if i not in none_indices],
                        [val for i, val in enumerate(row) if i not in none_indices])
                    model[corrname]['values'].append(corrval)

                model[corrname]['mean_simple'] = np.mean(model[corrname]['values'])
                model[corrname]['mean_weighted'] = \
                        np.average(model[corrname]['values'], weights=at)

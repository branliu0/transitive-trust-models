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

    self.global_ttms has the following format:
    {
        <global trust model>: {
            'scores': [<raw trust model score (1D)>],
            'pearson': <computed pearson score>,
            'kendalltau': <computed KT score>,
            'spearman': <computed spearman score>
        },
        ...
    }

    self.personalized_ttms has the following format:
    {
        <personalized trust model>: {
            'scores' [<raw trust model scores (2D)>],
            'pearson': {
                'values': [<pearson score for each agent (1D)],
                'mean_simple': <simple average of each agent's score>,
                'mean_weighted': <weighted average of each agent's score>
            },
            'kendalltau': ...,
            'spearman': ...
        },
        ...
    }

    self.info_scores has the following format:
    {
        'pearson': {
            'pagerank_weighted': ...,
            'hitting_time_weighted_all': ...,
            'hitting_time_weighted_top': ...,
            'max_flow': ...,
            'max_flow_weighted_means': ...,
            'shortest_path': ...,
            'shortest_path_weighted_means': ...,
        },
        'kendalltau': ...,
        'spearman': ...
    }
    """

    MODEL_NAMES = [
        'pagerank_weighted',
        'hitting_pagerank_all',
        'hitting_pagerank_top',
        'hitting_time_weighted_all',
        'hitting_time_weighted_top',
        'max_flow',
        'max_flow_weighted_means',
        'shortest_path',
        'shortest_path_weighted_means'
    ]

    CORRELATION_NAMES = ['pearson', 'kendalltau', 'spearman']

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
        self.info_scores = defaultdict(dict)
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
        self.global_ttms['hitting_time_weighted_top']['scores'] = \
                self.trust_models.hitting_time('top', weighted=True)
        self.global_ttms['hitting_pagerank_all']['scores'] = \
                self.trust_models.hitting_pagerank('all')
        self.global_ttms['hitting_pagerank_top']['scores'] = \
                self.trust_models.hitting_pagerank('top')

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
        for modelname, model in self.global_ttms.items():
            for corrname, corr in self.correlations.items():
                info_score, _ = corr(at, model['scores'])
                model[corrname] = info_score
                self.info_scores[corrname][modelname] = info_score

        for modelname, model in self.personalized_ttms.items():
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

                info_score_simple = np.mean(model[corrname]['values'])
                info_score_weighted = np.average(model[corrname]['values'],
                                                 weights=at)
                model[corrname]['mean_simple'] = info_score_simple
                model[corrname]['mean_weighted'] = info_score_weighted
                self.info_scores[corrname][modelname] = info_score_simple
                self.info_scores[corrname][modelname + '_weighted_means'] = \
                        info_score_weighted


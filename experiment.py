from collections import defaultdict
import time

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
            'pagerank': ...,
            'hitting_time_all': ...,
            'hitting_time_top': ...,
            'max_flow': ...,
            'max_flow_weighted_means': ...,
            'shortest_path': ...,
            'shortest_path_weighted_means': ...,
        },
        'kendalltau': ...,
        'spearman': ...
    }

    self.runtimes has the following format:
    {
        'pagerank_weighted': <float>
        ...
    }
    """

    # Format:
    # 1. True if a global TTM; False if a personalized TTM
    # 2. The name for the TTM; used as the key in the dict
    # 3. The name of the function on the TrustModel class
    # 4. (list) args to be passed to the ttm function
    # 5. (dict) kwargs to be passed to the ttm function
    TTM_PARAMS = [
        (True, 'pagerank', 'pagerank', [], {}),
        (True, 'hitting_time_all', 'hitting_time', ['all'], {}),
        (True, 'hitting_time_top', 'hitting_time', ['top'], {}),
        (False, 'max_flow', 'max_flow', [], {}),
        (False, 'shortest_path', 'shortest_path', [], {})
    ]

    MODEL_NAMES = [x[1] for x in TTM_PARAMS]

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
        self.runtimes = dict()

    def compute_informativeness(self):
        """ Compute trust model scores and measure informativeness. """
        self.compute_scores()
        self.measure_informativeness()

    def compute_scores(self):
        """ Actually run the trust model routines. Can take a while. """

        for is_global, name, model_method, args, kwargs in self.TTM_PARAMS:
            # First calculate the runtime of the method
            start_time = time.clock()
            score = getattr(self.trust_models, model_method)(*args, **kwargs)
            runtime = time.clock() - start_time

            # Then actually store it.
            d = self.global_ttms if is_global else self.personalized_ttms
            d[name]['scores'] = score
            self.runtimes[name] = runtime

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


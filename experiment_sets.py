from collections import defaultdict
import os
import time

import numpy as np
import yaml

from experiment import Experiment

SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "saved")

class EdgeCountExperimentSet(object):
    DEFAULT_EDGE_COUNTS = [2, 3, 4, 5, 10, 15, 20, 35, 49]  # for num_nodes = 50
    def __init__(self, num_nodes, agent_type_prior, edge_strategy,
                 edge_weight_strategy, num_weight_samples,
                 prefix, num_experiments, edge_counts=None):
        if not edge_counts:
            edge_counts = self.DEFAULT_EDGE_COUNTS

        self.num_nodes            = num_nodes
        self.agent_type_prior     = agent_type_prior
        self.edge_strategy        = edge_strategy
        self.edge_weight_strategy = edge_weight_strategy
        self.num_weight_samples   = num_weight_samples
        self.prefix               = prefix
        self.num_experiments      = num_experiments
        self.edge_counts          = (edge_counts if edge_counts
                                     else self.DEFAULT_EDGE_COUNTS)

        # Save attributes now, in case we want to recreate later.
        exp_filename = "%s_edge_count.yaml" % self.prefix
        with open(os.path.join(SAVE_FOLDER, exp_filename), 'w') as f:
            f.write(yaml.dump(self, indent=2))

        self.experiments = defaultdict(list)
        self.results = defaultdict(list)

    def run_experiments(self, clear=False):
        if clear:
            self.experiments = defaultdict(list)
            self.results = defaultdict(list)

        experiment_count = sum(len(x) for x in self.experiments)

        for edge_count in self.edge_counts:
            for _ in xrange(self.num_experiments - len(self.experiments[edge_count])):
                experiment_count += 1
                start_time = time.clock()

                exp = Experiment(self.num_nodes, self.agent_type_prior,
                                 self.edge_strategy, edge_count,
                                 self.edge_weight_strategy, self.num_weight_samples)
                exp.compute_informativeness()
                self.experiments[edge_count].append(exp)

                elapsed_time = time.clock() - start_time
                print "Experiment %d added in %0.2f seconds" % \
                        (experiment_count, elapsed_time)

                self.save_experiments()  # Always save progress!

    def gather_results(self):
        """
        Populates the self.results dict in the following format:
        {
            <correlation type>: {
                <transitive trust model>: {
                    <edge_count>: <average informativeness score>,
                    ...
                },
                ...
            },
            ...
        }

        - Value given by edge_count is a point on a line
        - Dict given by transitive trust model defines a line on a graph
        - Dict of dicts given by correlation type gives a set of lines for a graph
        - self.results provides a graph for each type of correlation measure.
        """
        self.results = {}
        for corrname in Experiment.CORRELATION_NAMES:
            self.results[corrname] = {}
            for modelname in Experiment.MODEL_NAMES:
                self.results[corrname][modelname] = {}
                for edge_count in self.edge_counts:
                    avg_score = np.mean([
                        exp.info_scores[corrname][modelname]
                        for exp in self.experiments[edge_count]])
                    self.results[corrname][modelname][edge_count] = avg_score

    def save_experiments(self):
        self._save(self.experiments, "experiments")

    def _save(self, obj, suffix=""):
        filename = "%s_edge_count%s.yaml" % (
            self.prefix, "_" + suffix if suffix else "")
        with open(os.path.join(SAVE_FOLDER, filename), 'w') as f:
            f.write(yaml.dump(obj, indent=2))

    @staticmethod
    def load_from_file(prefix):
        """ Deserialize a copy of an object from saved YAML files. """
        base_filename = os.path.join(SAVE_FOLDER, "%s_edge_count.yaml" % prefix)
        exp_filename = os.path.join(SAVE_FOLDER,
                                    "%s_edge_count_experiments.yaml" % prefix)
        results_filename = os.path.join(SAVE_FOLDER,
                                        "%s_edge_count_results.yaml" % prefix)

        if not os.path.exists(base_filename):
            raise ValueError("Save file does not exist.")

        with open(base_filename, 'r') as f:
            exp = yaml.load(f.read())

        if os.path.exists(exp_filename):
            with open(exp_filename, 'r') as f:
                exp.experiments = yaml.load(f.read())

        if os.path.exists(results_filename):
            with open(results_filename, 'r') as f:
                exp.results = yaml.load(f.read())

        return exp


class SampleCountExperimentSet(object):
    def __init__(self):
        pass

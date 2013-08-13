from collections import defaultdict
import os
import time

import yaml

from experiment import Experiment

SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "saved")

class EdgeCountExperimentSet(object):
    DEFAULT_EDGE_COUNTS = [1, 2, 3, 4, 5, 10, 15, 20, 35, 49]  # for num_nodes = 50
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

    def save_experiments(self):
        filename = "%s_edge_count_experiments.yaml" % self.prefix
        with open(os.path.join(SAVE_FOLDER, filename), 'w') as f:
            f.write(yaml.dump(self.experiments, indent=2))

    def load_experiments(self):
        filename = "%s_edge_count_experiments.yaml" % self.prefix
        with open(os.path.join(SAVE_FOLDER, filename), 'r') as f:
            self.experiments = yaml.load(f.read())

    @staticmethod
    def load_from_file(prefix):
        exp_filename = "%s_edge_count.yaml" % prefix
        with open(os.path.join(SAVE_FOLDER, exp_filename), 'r') as f:
            exp = yaml.load(f.read())
        exp.load_experiments()
        return exp


class SampleCountExperimentSet(object):
    def __init__(self):
        pass

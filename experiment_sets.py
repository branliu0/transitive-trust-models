from collections import defaultdict
import os
import time
import sys

import matplotlib.pyplot as plt
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

        if os.path.exists(self._filename()):
            raise ValueError("Experiment Set with this prefix already exists")

        # Save attributes now, for when we want to recreate later.
        self._save(self)

        self.experiments = defaultdict(list)
        self.results = defaultdict(list)

    def run_experiments(self, clear=False):
        if clear:
            self.experiments = defaultdict(list)
            self.results = defaultdict(list)

        experiment_count = sum(len(x) for x in self.experiments.values())

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

                self.save_experiment(exp, experiment_count)

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

        self._save(self.results, "results")

    def save_experiment(self, exp, num):
        exp_folder = os.path.join(SAVE_FOLDER,
                                  "%s_edge_count" % self.prefix)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder, 0755)

        exp_filename = os.path.join(exp_folder, "experiment.%03d.yaml" % num)
        if os.path.exists(exp_filename):
            print "Warning: would have overwritten file; backing up."
            new_filename = os.path.join(
                exp_folder, "experiment.%03d.%d.backup.yaml" % (num, time.time()))
            os.rename(exp_filename, new_filename)

        with open(exp_filename, 'w') as f:
            f.write(yaml.dump(exp, indent=2))

    def load_experiments(self):
        # TODO: Loading experiments is very slow. Consider paring down what gets
        # marshalled and saved?
        if (hasattr(self, "experiments") and
            isinstance(self.experiments, dict) and
            sum(len(x) for x in self.experiments.values()) != 0):
            raise ValueError("Error: self.experiments is populated. "
                             "Clear before loading.")

        exp_folder = os.path.join(SAVE_FOLDER,
                                  "%s_edge_count" % self.prefix)
        self.experiments = defaultdict(list)
        num_experiments = 0
        if os.path.exists(exp_folder):
            while True:
                filename = os.path.join(
                    exp_folder, "experiment.%03d.yaml" % (num_experiments + 1))
                if not os.path.exists(filename):
                    break

                with open(filename) as f:
                    exp = yaml.load(f.read())
                self.experiments[exp.graph.edges_per_node].append(exp)
                num_experiments += 1
                sys.stdout.write('.')

        print "%d experiments loaded" % num_experiments

    def _save(self, obj, suffix=""):
        with open(self._filename(suffix), 'w') as f:
            f.write(yaml.dump(obj, indent=2))

    def _filename(self, suffix=""):
        filename = "%s_edge_count%s.yaml" % (
            self.prefix, "_" + suffix if suffix else "")
        return os.path.join(SAVE_FOLDER, filename)

    @staticmethod
    def load_from_file(prefix, load_experiments=False):
        """ Deserialize a copy of an object from saved YAML files. """
        base_filename = os.path.join(SAVE_FOLDER, "%s_edge_count.yaml" % prefix)

        if not os.path.exists(base_filename):
            raise ValueError("Save file does not exist.")

        with open(base_filename, 'r') as f:
            exp_set = yaml.load(f.read())

        if load_experiments:
            exp_set.load_experiments()

        results_filename = exp_set._filename("results")
        if os.path.exists(results_filename):
            with open(results_filename, 'r') as f:
                exp_set.results = yaml.load(f.read())

        return exp_set

    def plot(self):
        PLOT_MARKERS = ['b--^', 'g--*', 'g--s', 'g--^',
                        'r--s', 'r--^', 'c--s', 'c--^']
        for corrname in Experiment.CORRELATION_NAMES:
            for i, modelname in enumerate(Experiment.MODEL_NAMES):
                points = sorted(self.results[corrname][modelname].items())
                plt.plot([x[0] for x in points], [x[1] for x in points],
                         PLOT_MARKERS[i], label=modelname)
            plt.suptitle('Varying edge count with graph of %d nodes' % self.num_nodes)
            plt.xlabel('Edges per node')
            plt.ylabel(corrname + ' correlation')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       fancybox=True, shadow=True)
            plt.show()

    def description(self):
        raise NotImplementedError


class SampleCountExperimentSet(object):
    def __init__(self):
        pass

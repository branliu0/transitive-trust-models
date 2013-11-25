from abc import ABCMeta
from collections import defaultdict
import math
import multiprocessing
import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml

from experiment import Experiment
from trust_graph import INFINITY

SAVE_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'saved')

def run_experiment(args):
    """ Parallelizable method for computing experiments.

    This method is used in parallel computation for running experiments in
    parallel. Due to the nature of pickling, it must be declared globally,
    because instance methods cannot be pickled.
    """
    # args is a tuple, so that we can map over an array of tuples.
    # see run_parallel_experiments()
    params, param_name, val = args
    start_time = time.clock()

    params = params.copy()
    params[param_name] = val

    while True:
        exp = Experiment(**params)
        try:
            exp.compute_informativeness()
            break
        except Exception, e:
            print str(e)

    elapsed_time = time.clock() - start_time
    print "Experiment with val %s added in %0.2f seconds" % \
            (str(val), elapsed_time)

    return val, exp


class ExperimentSet(object):
    """ Abstract base class used for forming experiment sets.

    Experiment sets are a collection of experiments. They primarily serve two
    purposes:
        1. They vary one independent parameter while holding all other
            parameters fixed.
        2. They run multiple iterations for a given set of parameters and takes
            the average of those results.

    The following class level attributes should be defined by subclasses:
        - name: The name given to the experiment set, e.g., edge_count. This
            name is used for saving files and ideally should not be changed
            because it is used to find file names.
    """
    __metaclass__ = ABCMeta

    def __init__(self, experiment_params, ind_param_name, ind_param_values,
                 prefix, num_experiments):
        required_fields = ['name', 'plot_title', 'plot_xlabel']
        for f in required_fields:
            if not getattr(self, f):
                raise ValueError("self.%s must be defined" % f)

        self.experiment_params = experiment_params
        self.ind_param_name    = ind_param_name
        self.ind_param_values  = ind_param_values
        self.prefix            = prefix
        self.num_experiments   = num_experiments
        for k, v in self.experiment_params.iteritems():
            setattr(self, k, v)
        setattr(self, ind_param_name, ind_param_values)

        if os.path.exists(self._filename()):
            raise ValueError("Experiment Set with this prefix already exists")

        # Save attributes now, for when we want to recreate later.
        self._save(self)

        self.experiments = defaultdict(list)

    ###################################
    # Functions related to computation
    ###################################

    def run_parallel_experiments(self):
        # NOTE: Always re-runs all the experiments.
        self.experiments = defaultdict(list)

        if not hasattr(self, 'failed_experiments'):
            self.failed_experiments = []

        vals = np.array([np.repeat(val, self.num_experiments)
                         for val in self.ind_param_values]).flatten()
        args = [(self.experiment_params, self.ind_param_name, v) for v in vals]

        pool = multiprocessing.Pool(processes=4)
        results = pool.map(run_experiment, args)
        pool.close()
        pool.join()
        for val, exp in results:
            self.experiments[val].append(exp)

        self.aggregate_results()
        self.aggregate_runtimes()

    def run_experiments(self, clear=False):
        if clear:
            self.experiments = defaultdict(list)

        if not hasattr(self, 'failed_experiments'):
            self.failed_experiments = []

        experiment_count = sum(len(x) for x in self.experiments.values())
        params = self.experiment_params.copy()

        for val in self.ind_param_values:
            for _ in xrange(self.num_experiments - len(self.experiments[val])):
                experiment_count += 1
                start_time = time.clock()

                params[self.ind_param_name] = val

                # Sometimes running experiments throws exceptions -- mainly
                # max flow for some as of now unknown reason.
                # We could possibly be concerned about slight biasing because
                # we're not getting an unbiased distribution over graphs, but
                # this seems to happen rarely enough that it isn't a problem.
                while True:
                    exp = Experiment(**params)
                    try:
                        exp.compute_informativeness()
                        self.experiments[val].append(exp)
                        break
                    except Exception, e:
                        self.failed_experiments.append(exp)
                        print str(e)

                elapsed_time = time.clock() - start_time
                print "Experiment %d added in %0.2f seconds" % \
                        (experiment_count, elapsed_time)

                # self.save_experiment(exp, experiment_count)

        self.aggregate_results()
        self.aggregate_runtimes()

    def aggregate_results(self):
        """ Aggregates results from all the experiments.

        Populates the self.results dict in the following format:
        {
            <correlation type>: {
                <transitive trust model>: {
                    <param_val>: <average informativeness score>,
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
        self.results = {}; self.errors = {}
        for corrname in Experiment.CORRELATIONS.keys():
            self.results[corrname] = {}; self.errors[corrname] = {}
            for modelname in Experiment.MODEL_NAMES:
                self.results[corrname][modelname] = {}
                self.errors[corrname][modelname] = {}
                for val in self.ind_param_values:
                    scores = [exp.info_scores[corrname][modelname]
                              for exp in self.experiments[val]]
                    self.results[corrname][modelname][val] = np.mean(scores)
                    # 95% Confidence Intervals, assuming normality.
                    self.errors[corrname][modelname][val] = 1.96 * stats.sem(scores)

        self._save(self.errors, "errors")
        self._save(self.results, "results")

    def aggregate_runtimes(self):
        """ Aggregates te runtime results from all the experiments.

        Populates the self.runtimes dict in the following format:
        {
            <transitive trust model>: {
                <param_val>: <average runtime>,
                ...
            },
            ...
        }
        """
        self.runtimes = {}
        for modelname in Experiment.MODEL_NAMES:
            self.runtimes[modelname] = {}
            for val in self.ind_param_values:
                avg_runtime = np.mean([exp.runtimes[modelname]
                                       for exp in self.experiments[val]])
                self.runtimes[modelname][val] = avg_runtime

        self._save(self.runtimes, "runtimes")

    #################################################
    # Functions related to display and visualization
    #################################################

    PLOT_MARKERS = {
        'pagerank': 'b--^',
        'pagerank_weighted': 'b--^',  # Backwards compatibility
        'hitting_pagerank_all': 'g--*',
        'hitting_pagerank_top': 'g--^',
        'hitting_time_all': 'm--*',
        'hitting_time_weighted_all': 'm--*',  # Backwards compatibility
        'hitting_time_top': 'm--^',
        'hitting_time_weighted_top': 'm--^',  # Backwards compatibility
        'max_flow': 'r--s',
        'max_flow_weighted_means': 'r--^',
        'shortest_path': 'c--s',
        'shortest_path_weighted_means': 'c--^'
    }

    def transform_x(self, xs):
        """ Function to transform the x-axis (e.g., use a log scale)

        Returns:
            A tuple (xvals, xticks) where xvals are the actual x-values to be
            used for graphing, and xticks, if non-empty, are strings to be
            used for x-axis ticks.
        """
        return xs, xs

    def plot(self, filename=None):
        extra_artists = []
        n = len(Experiment.CORRELATIONS)
        for i, corrname in enumerate(Experiment.CORRELATIONS.keys()):
            plt.subplot(n, 1, i + 1)
            for modelname in Experiment.MODEL_NAMES:
                points = sorted(self.results[corrname][modelname].items())
                xvals, xticks = self.transform_x([x[0] for x in points])
                yerrs = [x[1] for x in sorted(
                    self.errors[corrname][modelname].items())]
                plt.errorbar(xvals, [x[1] for x in points], yerrs,
                         fmt=self.PLOT_MARKERS[modelname], label=modelname)
                if xticks:
                    plt.xticks(xvals, xticks)
            plt.xlabel(self.plot_xlabel)
            plt.margins(0.1)
            plt.ylabel(corrname + ' correlation')
            extra_artists.append(
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                           fancybox=True, shadow=True))

        fig = plt.gcf()
        # Sadly hard-coding this in for now...
        fig.set_figheight(n * 4.00)
        fig.set_figwidth(8.40)
        extra_artists.append(fig.suptitle(self.plot_title))  # Add plot title

        if filename and isinstance(filename, str):
            # Need to specify the extra artists so that they show up in the
            # saved image. Calling bbox_inches='tight' makes it calculate the
            # correct bounds for the image.
            plt.savefig(filename, bbox_extra_artists=extra_artists,
                        bbox_inches='tight')
            # Clear figure, or this figure gets re-painted by subsequent calls
            plt.clf()
        else:
            plt.show()

    def plot_runtimes(self):
        for modelname in Experiment.MODEL_NAMES:
            points = sorted(self.runtimes[modelname].items())
            plt.plot(self.transform_x([x[0] for x in points]),
                     [x[1] for x in points],
                     self.PLOT_MARKERS[modelname], label=modelname)
        plt.suptitle("Runtimes for transitive trust models")
        plt.xlabel(self.plot_xlabel)
        plt.ylabel("Average Runtime (sec)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   fancybox=True, shadow=True)
        plt.show()

    def description(self):
        return """\
num_nodes            = {num_nodes}
agent_type_prior     = {agent_type_prior}
edge_strategy        = {edge_strategy}
edges_per_node       = {edges_per_node}
edge_weight_strategy = {edge_weight_strategy}
num_weight_samples   = {num_weight_samples}
prefix               = {prefix}
num_experiments      = {num_experiments}""".format(**self.__dict__)

    ##########################################################
    # Functions related to saving and loading from YAML files
    ##########################################################

    @classmethod
    def load_from_file(cls, prefix, load_experiments=False):
        """ Retrive and load an experiment set from YAML files. """
        base_filename = os.path.join(
            SAVE_FOLDER, "%s_%s.yaml" % (prefix, cls.name))

        if not os.path.exists(base_filename):
            raise ValueError("Save file does not exist.")

        with open(base_filename, 'r') as f:
            exp_set = yaml.load(f.read())

        exp_set.experiments = defaultdict(list)

        if load_experiments:
            exp_set.load_experiments()

        for prop in ["results", "errors", "runtimes"]:
            if os.path.exists(exp_set._filename(prop)):
                setattr(exp_set, prop, exp_set._load(prop))

        return exp_set

    def save_experiment(self, exp, num):
        """ Saves an experiment to disk. """
        exp_folder = os.path.join(SAVE_FOLDER, "%s_%s" % (self.prefix, self.name))
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder, 0755)

        exp_filename = os.path.join(exp_folder, "experiment.%03d.yaml" % num)
        if os.path.exists(exp_filename):
            print "Warning: would have overwritten file; backing up."
            new_filename = os.path.join(
                exp_folder, "experiment.%03d.%d.backup.yaml" % (num, time.time()))
            os.rename(exp_filename, new_filename)

        self._save(exp, filename=exp_filename)

    def load_experiments(self):
        """ Loads all experiments for this experiment set into memory. """
        # TODO: Loading experiments is very slow. Consider paring down what gets
        # marshalled and saved? Or consider using a DB.
        print "Warning: This function currently is not optimized and is slow."
        if (hasattr(self, "experiments") and
            isinstance(self.experiments, dict) and
            sum(len(x) for x in self.experiments.values()) != 0):
            raise ValueError("Error: self.experiments is populated. "
                             "Clear before loading to avoid overwriting.")

        exp_folder = os.path.join(SAVE_FOLDER, "%s_%s" % (self.prefix, self.name))
        self.experiments = defaultdict(list)
        num_experiments = 0
        if os.path.exists(exp_folder):
            while True:
                filename = os.path.join(
                    exp_folder, "experiment.%03d.yaml" % (num_experiments + 1))
                if not os.path.exists(filename):
                    break

                exp = self._load(filename=filename)
                self.experiments[exp.graph.edges_per_node].append(exp)
                num_experiments += 1
                sys.stdout.write('.')

        print "%d experiments loaded" % num_experiments

    def _filename(self, suffix=""):
        """ Generates a filename used for saving or loading files. """
        filename = "{prefix}_{name}{suffix}.yaml".format(
            prefix=self.prefix, name=self.name,
            suffix=("_" + suffix if suffix else ""))
        return os.path.join(SAVE_FOLDER, filename)

    def _save(self, obj, suffix="", filename=""):
        """ General purpose function used for saving objects as YAML files. """
        if not filename:
            filename = self._filename(suffix)
        with open(filename, 'w') as f:
            f.write(yaml.dump(obj, indent=2))

    def _load(self, suffix="", filename=""):
        """ General purpose function used for loading objects from YAML. """
        if not filename:
            filename = self._filename(suffix)
        with open(filename, 'r') as f:
            return yaml.load(f.read())

class EdgeCountExperimentSet(ExperimentSet):
    name = 'edge_count'
    plot_xlabel = 'Edges per node'

    DEFAULT_EDGE_COUNTS = [2, 3, 4, 5, 10, 15, 20, 35, 49]  # for num_nodes = 50

    def __init__(self, num_nodes, agent_type_prior, edge_strategy,
                 edge_weight_strategy, num_weight_samples,
                 prefix, num_experiments, edge_counts=None):
        """
        Args:
            num_nodes: Number of nodes in this graph.
            agent_type_prior:
                'uniform': Selected from Unif[0, 1]
                'normal': Selected from Normal[0.5, 1] truncated to [0, 1]
                'beta': Selected from Beta[2, 2]
            edge_strategy:
                'uniform': Neighbors are uniformly selected
                'cluster': High types are more likely to connect to high types
            edge_weight_strategy:
                'sample': Sample from true agent type
                'noisy': Low types more likely to sample from Bernoulli[0.5]
                'prior': Low types more likely to sampel from prior distribution
            num_weight_samples: Number of times to sample for determining
                edge weights.
            prefix: Prefix used for saving
            num_experiments: Number of experiments to run per parameter set
            edge_counts: An array of edge counts to vary over
        """
        if not edge_counts:
            edge_counts = self.DEFAULT_EDGE_COUNTS

        params = {
            'num_nodes': num_nodes,
            'agent_type_prior': agent_type_prior,
            'edge_strategy': edge_strategy,
            'edge_weight_strategy': edge_weight_strategy,
            'num_weight_samples': num_weight_samples
        }

        self.plot_title = (
            "TTM Informativeness: Varying number of edges per node\n"
            "%d nodes, '%s' prior, '%s' edges, '%s' weights (%d samples) (n = %d)"
            % (num_nodes, agent_type_prior, edge_strategy, edge_weight_strategy,
               num_weight_samples, num_experiments))

        super(EdgeCountExperimentSet, self).__init__(
            params, 'edges_per_node', edge_counts, prefix, num_experiments)


class SampleCountExperimentSet(ExperimentSet):
    name = 'sample_count'
    plot_xlabel = 'log2(samples per edge)'

    DEFAULT_SAMPLE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, INFINITY]

    def __init__(self, num_nodes, agent_type_prior, edge_strategy,
                 edges_per_node, edge_weight_strategy, prefix,
                 num_experiments, sample_counts=None):
        """
        Args:
            num_nodes: Number of nodes in this graph.
            agent_type_prior:
                'uniform': Selected from Unif[0, 1]
                'normal': Selected from Normal[0.5, 1] truncated to [0, 1]
                'beta': Selected from Beta[2, 2]
            edge_strategy:
                'uniform': Neighbors are uniformly selected
                'cluster': High types are more likely to connect to high types
            edges_per_node: The number of outgoing edges each node has.
            edge_weight_strategy:
                'sample': Sample from true agent type
                'noisy': Low types more likely to sample from Bernoulli[0.5]
                'prior': Low types more likely to sampel from prior distribution
            prefix: Prefix used for saving
            num_experiments: Number of experiments to run per parameter set
            sample_counts: An array of sample counts to vary over
        """
        if not sample_counts:
            sample_counts = self.DEFAULT_SAMPLE_COUNTS

        params = {
            'num_nodes': num_nodes,
            'agent_type_prior': agent_type_prior,
            'edge_strategy': edge_strategy,
            'edges_per_node': edges_per_node,
            'edge_weight_strategy': edge_weight_strategy,
        }

        self.plot_title = (
            "TTM Informativeness: Varying number of weight samples per edge\n"
            "%d nodes/%d edges per node, '%s' prior, '%s' edges, '%s' weights (n = %d)"
            % (num_nodes, edges_per_node, agent_type_prior, edge_strategy,
               edge_weight_strategy, num_experiments))

        super(SampleCountExperimentSet, self).__init__(
            params, 'num_weight_samples', sample_counts, prefix, num_experiments)

    def transform_x(self, xs):
        """ Use a log-2 scale and handle values of infinity. """
        # Values of infinity are set as 3 ticks higher than the max value
        INF_OFFSET = 3
        transformed = [math.log(x, 2) for x in xs if x != INFINITY]
        ticks = list(transformed)
        try:
            inf_idx = next(i for i, x in enumerate(xs) if x == INFINITY)
            transformed.insert(inf_idx, max(transformed) + INF_OFFSET)
            ticks.insert(inf_idx, 'infinity')
        except StopIteration:
            pass
        return transformed, ticks

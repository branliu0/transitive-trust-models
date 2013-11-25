import os

from experiment_sets import EdgeCountExperimentSet
from experiment_sets import SampleCountExperimentSet
from experiment_sets import SAVE_FOLDER
from trust_graph import TrustGraph

NUM_NODES = 50
DEFAULT_EDGE_COUNT = 15
DEFAULT_SAMPLE_COUNT = 30

def run_all_experiments(num_experiments, folder_name, num_processes=None):
    folder = os.path.join(SAVE_FOLDER, folder_name)
    os.mkdir(folder)

    count = 1

    for prior in TrustGraph.AGENT_TYPE_PRIORS:
        for edge_strat in TrustGraph.EDGE_STRATEGIES:
            for weight_strat in TrustGraph.EDGE_WEIGHT_STRATEGIES:
                # Edge Count Experiment Sets
                eces = EdgeCountExperimentSet(
                    NUM_NODES, prior, edge_strat, weight_strat,
                    DEFAULT_SAMPLE_COUNT, "%s_%d" % (folder_name, count),
                    num_experiments)
                eces.run_parallel_experiments(num_processes)
                filename = ("edges_%d_%s_%s_%s_%dt_%de.png"
                            % (NUM_NODES, prior, edge_strat, weight_strat,
                               DEFAULT_SAMPLE_COUNT, num_experiments))
                eces.plot(os.path.join(folder, filename))

                # Sample Count Experiment Sets
                sces = SampleCountExperimentSet(
                    NUM_NODES, prior, edge_strat, DEFAULT_EDGE_COUNT,
                    weight_strat, "%s_%d" % (folder_name, count),
                    num_experiments)
                sces.run_parallel_experiments(num_processes)
                filename = ("samples_%d-%d_%s_%s_%s_%de.png"
                            % (NUM_NODES, DEFAULT_EDGE_COUNT, prior,
                               edge_strat, weight_strat, num_experiments))
                sces.plot(os.path.join(folder, filename))

                count += 1

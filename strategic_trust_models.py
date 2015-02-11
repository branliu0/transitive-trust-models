""" Strategic Trust Models

This file provides routines that compute various trust model under strategic
manipulation. A given number of agents are chosen to be strategic, and they
perform optimal manipulations to make themselves look better, and then the
trust model scores are applied.
"""
import random
import sys

import networkx as nx
import numpy as np
from scipy import stats

from hitting_time.mat_hitting_time import single_LS_prob_ht
import trust_models as tm
import utils

NUM_SMC_TRIALS = 10


def random_strategic_agents(graph, num_strategic):
    return random.sample(graph.nodes(), num_strategic)


def lowtype_strategic_agents(graph, num_strategic):
    probs = 1 - np.array(graph.agent_types)
    indices = np.arange(graph.number_of_nodes())
    strategic_agents = []
    for _ in xrange(num_strategic):
        rv = stats.rv_discrete(name='lowtype', values=(
            [i for i in indices if i not in strategic_agents],
            utils.normalize([val for i, val in enumerate(probs)
                             if i not in strategic_agents])))
        strategic_agents.append(rv.rvs())
    return np.array(graph.nodes())[strategic_agents]

RANDOM_PRIME = 97


def cut_outlinks(graph, agents, keep_zero=False, leave_one=False):
    if keep_zero:
        for u, v in graph.edges(agents):
            graph[u][v]['weight'] = 0
    # elif leave_one:
        # for a in agents:
            # edges = graph.edges(a)
            # # Try to be deterministic about the edge that is removed lol.
            # removed_index = RANDOM_PRIME % len(edges)
            # removed_edges = edges[:removed_index] + edges[(removed_index + 1):]
            # graph.remove_edges_from(removed_edges)
            # # graph.remove_edges_from(random.sample(edges, len(edges) - 1))
    else:
        graph.remove_edges_from(graph.edges(agents))

IDEAL_RADIUS = 3  # Arbitrarily picked...
SYBIL_WEIGHT = 1  # We're keeping weights in [0, 1] now.


def generate_sybils(graph, agents, num_sybils, randomize_sybils=True,
                    sybil_radius=IDEAL_RADIUS, sybil_cloud=False,
                    sybil_star=False):
    """
    Randomizes the number of sybils so that not all the agents have the same
    number of sybils.
    """
    if num_sybils <= 0:
        return

    sybil_counter = max(graph.nodes()) + 1

    # Parametrize the uniform distribution for sybil counts
    max_radius = num_sybils - 1  # distance from
    radius = min(max_radius, sybil_radius)
    unif_a = num_sybils - radius
    unif_b = num_sybils + radius

    for agent in agents:
        sybil_count = random.randint(unif_a, unif_b) if randomize_sybils else num_sybils
        edges = [(agent, sybil)
                 for sybil in xrange(sybil_counter, sybil_counter + sybil_count)]
        if not sybil_star:
            edges += map(lambda x: x[::-1], edges)  # Add reverse edges
        if sybil_cloud:
            edges += [(sybil_counter + i, sybil_counter + j)
                      for i in xrange(sybil_count) for j in xrange(sybil_count)
                      if i != j]
        graph.add_edges_from(edges, weight=SYBIL_WEIGHT, inv_weight=1/float(SYBIL_WEIGHT))
        sybil_counter += sybil_count


# WARNING: This is left here for compatibility purposes, but this should not
# be used, as it messes up calculations!
# THIN_EDGE_WEIGHT=1e-5

# def add_thin_edges(graph, edge_weight=THIN_EDGE_WEIGHT):
    # edges = graph.edges()
    # for i in graph.nodes_iter():
        # for j in graph.nodes_iter():
            # if i == j:
                # continue
            # if (i, j) not in edges:
                # graph.add_edge(i, j, weight=edge_weight,
                               # inv_weight=(1 / edge_weight))



def global_pagerank(graph, num_strategic, sybil_pct, cutlinks=True,
                    gensybils=True, strategic_agents=None,
                    return_data=False):
    """ Global PageRank.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    N = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    if sybil_pct is None:
        sybil_pct = 0.3
    num_sybils = int(sybil_pct * N)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)
    scores = tm.pagerank(graph)[:N]

    if return_data:
        return scores, {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return scores


def person_pagerank(graph, num_strategic, sybil_pct,
                    cutlinks=True, gensybils=True, strategic_agents=None,
                    return_data=False):
    """ Personalized PageRank """
    graph = graph.copy()
    origN = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    if sybil_pct is None:
        sybil_pct = 0.3
    if cutlinks:
        saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
        cut_outlinks(graph, strategic_agents, leave_one=True)
        after_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
    if gensybils:
        generate_sybils(graph, strategic_agents, int(sybil_pct * origN))

    # We manually loop through personalize PageRank ourselves so that we can
    # avoid computing it for sybils.
    N = graph.number_of_nodes()
    scores = np.zeros((origN, N))
    personalization = {n: 0 for n in graph.nodes()}
    for i in xrange(origN):  # No need to compute for sybils
        # Go back to old edges
        # if cutlinks and i in strategic_agents:
            # graph.remove_edges_from(graph.edges(i))
            # graph.add_edges_from(saved_edges[i])

        personalization[i] = 1
        scores[i] = nx.pagerank_numpy(graph, personalization=personalization,
                                      weight='weight').values()
        personalization[i] = 0

        # Restore post-cut edges
        # if cutlinks and i in strategic_agents:
            # graph.remove_edges_from(graph.edges(i))
            # graph.add_edges_from(after_edges[i])


    if return_data:
        return scores[:origN, :origN], {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return scores[:origN, :origN]


def global_hitting_time(graph, num_strategic, sybil_pct, cutlinks=True,
                        gensybils=True, strategic_agents=None,
                        return_data=False):
    graph = graph.copy()
    origN = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    if sybil_pct is None:
        sybil_pct = 0.3
    num_sybils = int(origN * sybil_pct)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)

    ht = np.zeros(origN)
    for j in xrange(origN):
        # Adding is the same as applying a uniform restart distribution over
        # all nodes, including sybils
        # We negate to correct for the direction of correlation
        ht[j] = np.sum(single_LS_prob_ht(graph, j))

    if return_data:
        return ht, {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return ht


def person_hitting_time(graph, num_strategic, sybil_pct, cutlinks=True,
                        gensybils=True, strategic_agents=None,
                        return_data=False):
    graph = graph.copy()
    origN = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    if sybil_pct is None:
        sybil_pct = 0.3
    if cutlinks:
        cut_outlinks(graph, strategic_agents, leave_one=True)
    if gensybils:
        generate_sybils(graph, strategic_agents, int(sybil_pct * origN),
                        sybil_star=True)

    N = graph.number_of_nodes()
    ht = np.zeros((N, origN))
    for j in xrange(origN):
        # NOTE: WE SHOULD be adding back edges here.
        # BUT, because we're starting to average only over non-strategic agents,
        # I'm not going to bother implementing it for now.
        ht[:, j] = single_LS_prob_ht(graph, j)

    if return_data:
        return ht[:origN, :origN], {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return ht[:origN, :origN]


def person_max_flow(graph, num_strategic, sybil_pct=0, cutlinks=True,
                    gensybils=True, strategic_agents=None,
                    return_data=False):
    graph = graph.copy()
    N = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    saved_edges = {}
    if cutlinks:
        saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
        cut_outlinks(graph, strategic_agents, leave_one=True)
        after_edges = {a: graph.edges(a, data=True) for a in strategic_agents}

    # For Max Flow, don't apply sybils since it is strategyproof to sybils
    # if gensybils:
        # num_sybils = int(graph.number_of_nodes() * sybil_pct)
        # generate_sybils(graph, strategic_agents, num_sybils)

    # Need to reimplement max flow here because we only want to cut outedges
    # When we're not being evaluated.
    scores = np.zeros((N, N))
    for i in xrange(N):
        # Add back in the edges for this agent, so we can get an actual score.
        # if cutlinks and i in strategic_agents:
            # graph.remove_edges_from(graph.edges(i))
            # graph.add_edges_from(saved_edges[i])

        # Now compute the max flow scores
        for j in xrange(N):
            if i != j:
                scores[i, j] = nx.maximum_flow_value(graph, i, j, capacity='weight')
                # scores[i, j] = utils.fast_max_flow(graph.gt_graph, i, j)

        # Restore post-cut edges
        # if cutlinks and i in strategic_agents:
            # graph.remove_edges_from(graph.edges(i))
            # graph.add_edges_from(after_edges[i])

        sys.stdout.write('.')
    sys.stdout.write("\n")

    if return_data:
        return scores, {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return scores


def person_shortest_path(graph, num_strategic, sybil_pct=0, cutlinks=True,
                         gensybils=True, strategic_agents=None,
                         return_data=False):
    # For shortest path, we're not going to bother applying any manipulations,
    # because shortest path is strategyproof to all of them.

    # graph = graph.copy()
    origN = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    # num_sybils = int(graph.number_of_nodes() * sybil_pct)
    # saved_edges = {}
    # if cutlinks:
        # saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
        # cut_outlinks(graph, strategic_agents)
    # if gensybils:
        # generate_sybils(graph, strategic_agents, num_sybils)

    shortest_paths = np.zeros((origN, origN))
    for i in xrange(origN):
        # Add back in outedges
        # if cutlinks and i in saved_edges:
            # graph.add_edges_from(saved_edges[i])

        paths = nx.single_source_dijkstra_path_length(
            graph, i, weight='inv_weight')
        for j in xrange(origN):
            try:
                shortest_paths[i, j] = 1 / paths[j]
            except ZeroDivisionError:
                shortest_paths[i, j] = None
            except KeyError:  # Means i is not connected to j?
                shortest_paths[i, j] = 0  # Worst possible score

        # Cut those outlinks again
        # cut_outlinks(graph, i)

    if return_data:
        return shortest_paths, {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return shortest_paths


def average_ratings(graph, num_strategic, sybil_pct, cutlinks=True,
                    gensybils=True, strategic_agents=None,
                    return_data=False):
    graph = graph.copy()
    N = graph.number_of_nodes()
    if strategic_agents is None:
        strategic_agents = random_strategic_agents(graph, num_strategic)
    if sybil_pct is None:
        sybil_pct = 5
    Ns = int(sybil_pct * N)
    if cutlinks:
        cut_outlinks(graph, strategic_agents, keep_zero=True)
    if gensybils:
        generate_sybils(graph, strategic_agents, Ns)

    scores = tm.average_ratings(graph)[:N]

    if return_data:
        return scores, {'strategic_agents': strategic_agents, 'graph': graph}
    else:
        return scores

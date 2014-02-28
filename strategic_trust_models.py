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

from hitting_time.mat_hitting_time import single_LA_ht
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


def cut_outlinks(graph, agents):
    graph.remove_edges_from(graph.edges(agents))

IDEAL_RADIUS = 3  # Arbitrarily picked...

def generate_sybils(graph, agents, num_sybils, randomize_sybils=True):
    """
    Randomizes the number of sybils so that not all the agents have the same
    number of sybils.
    """
    sybil_counter = max(graph.nodes()) + 1

    # Parametrize the uniform distribution for sybil counts
    max_radius = num_sybils - 1  # distance from
    radius = min(max_radius, IDEAL_RADIUS)
    unif_a = num_sybils - radius
    unif_b = num_sybils + radius


    for agent in agents:
        sybil_count = random.randint(unif_a, unif_b) if randomize_sybils else num_sybils
        edges = [(agent, sybil)
                 for sybil in xrange(sybil_counter, sybil_counter + sybil_count)]
        edges += map(lambda x: x[::-1], edges)  # Add reverse edges
        graph.add_edges_from(edges)
        sybil_counter += sybil_count


THIN_EDGE_WEIGHT=1e-5

def add_thin_edges(graph, edge_weight=THIN_EDGE_WEIGHT):
    edges = graph.edges()
    for i in graph.nodes_iter():
        for j in graph.nodes_iter():
            if i == j:
                continue
            if (i, j) not in edges:
                graph.add_edge(i, j, weight=edge_weight,
                               inv_weight=(1 / edge_weight))


def global_pagerank(graph, num_strategic, sybil_pct,
                    cutlinks=True, gensybils=True):
    """ Global PageRank.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = random_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)
    return tm.pagerank(graph)[:N]


def person_pagerank(graph, num_strategic, sybil_pct,
                    cutlinks=True, gensybils=True):
    """ Personalized PageRank.

    Cut all outlinks + generate one sybil.
    """
    graph = graph.copy()
    origN = graph.number_of_nodes()

    strategic_agents = random_strategic_agents(graph, num_strategic)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
        add_thin_edges(graph)  # do this BEFORE sybils!!
    if gensybils:
        generate_sybils(graph, strategic_agents, min(1, sybil_pct * origN))

    # We manually loop through personalize PageRank ourselves so that we can
    # avoid computing it for sybils.
    N = graph.number_of_nodes()
    scores = np.zeros((origN, N))
    personalization = {n: 0 for n in graph.nodes()}
    for i in xrange(origN):  # No need to compute for sybils
        personalization[i] = 1
        scores[i] = nx.pagerank_numpy(graph, personalization=personalization,
                                      weight='weight').values()
        personalization[i] = 0

    return scores[:origN, :origN]


def global_hitting_time(graph, num_strategic, sybil_pct,
                        cutlinks=True, gensybils=True):
    graph = graph.copy()
    origN = graph.number_of_nodes()
    strategic_agents = random_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
        add_thin_edges(graph)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)

    ht = np.zeros(origN)
    for j in xrange(origN):
        # Adding is the same as applying a uniform restart distribution over
        # all nodes, including sybils
        # We negate to correct for the direction of correlation
        ht[j] = np.sum(single_LA_ht(graph, j))

    return ht


def person_hitting_time(graph, num_strategic, sybil_pct,
                        cutlinks=True, gensybils=True):
    graph = graph.copy()
    origN = graph.number_of_nodes()
    strategic_agents = random_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    if cutlinks:
        cut_outlinks(graph, strategic_agents)
        add_thin_edges(graph)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)

    N = graph.number_of_nodes()
    ht = np.zeros((N, origN))
    for j in xrange(origN):
        ht[:, j] = single_LA_ht(graph, j)

    return ht[:origN, :origN]


def person_max_flow(graph, num_strategic, sybil_pct,
                    cutlinks=True, gensybils=True):
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = random_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    saved_edges = {}
    if cutlinks:
        saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
        cut_outlinks(graph, strategic_agents)
        add_thin_edges(graph)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)

    # Need to reimplement max flow here because we only want to cut outedges
    # When we're not being evaluated.
    scores = np.zeros((N, N))
    for i in xrange(N):
        # Add back in the edges for this agent, so we can get an actual score.
        if i in saved_edges:
            for a, b, d in saved_edges[i]:
                graph[a][b]['weight'] = d['weight']

        # Now compute the max flow scores
        for j in xrange(N):
            if i == j:
                scores[i][j] = None
            else:
                mf = nx.max_flow(graph, i, j, capacity='weight')
                scores[i][j] = None if mf == 0 else mf

        # Now remove those edges again (a bit inefficiently)
        if i in saved_edges:
            for a, b, _ in saved_edges[i]:
                graph[a][b]['weight'] = THIN_EDGE_WEIGHT

        sys.stdout.write('.')
    sys.stdout.write("\n")

    return scores


def person_shortest_path(graph, num_strategic, sybil_pct,
                         cutlinks=True, gensybils=True):
    graph = graph.copy()
    origN = graph.number_of_nodes()
    strategic_agents = random_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    saved_edges = {}
    if cutlinks:
        saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
        cut_outlinks(graph, strategic_agents)
        add_thin_edges(graph)
    if gensybils:
        generate_sybils(graph, strategic_agents, num_sybils)

    shortest_paths = np.zeros((origN, origN))
    for i in xrange(origN):
        # Add back in outedges
        if i in saved_edges:
            for a, b, d in saved_edges[i]:
                graph[a][b]['inv_weight'] = d['inv_weight']

        paths = nx.single_source_dijkstra_path_length(
            graph, i, weight='inv_weight')
        for j in xrange(origN):
            try:
                shortest_paths[i, j] = 1 / paths[j]
            except ZeroDivisionError:
                shortest_paths[i, j] = None

        # remove them again
        if i in saved_edges:
            for a, b, _ in saved_edges[i]:
                graph[a][b]['inv_weight'] = 1 / THIN_EDGE_WEIGHT

    return shortest_paths

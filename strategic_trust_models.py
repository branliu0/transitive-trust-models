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

from hitting_time.single_monte_carlo import complete_path_smc_hitting_time
from hitting_time.mat_hitting_time import personalized_LA_ht
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


def generate_sybils(graph, agents, num_sybils):
    sybil_counter = max(graph.nodes()) + 1
    for agent in agents:
        edges = [(agent, sybil)
                 for sybil in xrange(sybil_counter, sybil_counter + num_sybils)]
        edges += map(lambda x: x[::-1], edges)
        graph.add_edges_from(edges)
        sybil_counter += num_sybils


THIN_EDGE_WEIGHT=1e-5

def add_thin_edges(graph, edge_weight=THIN_EDGE_WEIGHT):
    edges = graph.edges()
    for i in graph.nodes_iter():
        for j in graph.nodes_iter():
            if i == j:
                continue
            if (i, j) not in edges:
                graph.add_edge(i, j, weight=edge_weight)
                # We don't bother with inv_weight because this method doesn't
                # need to be used for shortest path.


def global_pagerank(graph, num_strategic, sybil_pct):
    """ Global PageRank.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, num_sybils)
    return tm.pagerank(graph)[:N]


def person_pagerank(graph, num_strategic, sybil_pct):
    """ Personalized PageRank.

    Cut all outlinks + generate one sybil.
    """
    graph = graph.copy()
    origN = graph.number_of_nodes()

    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    cut_outlinks(graph, strategic_agents)
    add_thin_edges(graph)  # do this BEFORE sybils!!
    generate_sybils(graph, strategic_agents, 1)

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


def global_hitting_time(graph, num_strategic, sybil_pct):
    """ Global Hitting Time.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, num_sybils)
    return tm.hitting_pagerank(graph, 'all')[:N]


def global_smc_hitting_time(graph, num_strategic, sybil_pct):
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, num_sybils)

    ht = complete_path_smc_hitting_time(graph, NUM_SMC_TRIALS)
    return ht.sum(axis=1)[:N]


def person_hitting_time(graph, num_strategic, sybil_pct):
    """ Personalized Hitting Time.

    Cut all outlinks.
    """
    graph = graph.copy()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    cut_outlinks(graph, strategic_agents)
    add_thin_edges(graph)
    return personalized_LA_ht(graph)


def person_smc_hitting_time(graph, num_strategic, sybil_pct):
    graph = graph.copy()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    cut_outlinks(graph, strategic_agents)
    add_thin_edges(graph)
    return complete_path_smc_hitting_time(graph, NUM_SMC_TRIALS)


def person_max_flow(graph, num_strategic, sybil_pct):
    """ Personalized Max Flow.

    Cut all outlinks.
    """
    graph = graph.copy()
    N = graph.number_of_nodes()
    strategic_agents = lowtype_strategic_agents(graph, num_strategic)
    saved_edges = {a: graph.edges(a, data=True) for a in strategic_agents}
    cut_outlinks(graph, strategic_agents)
    add_thin_edges(graph)

    # Need to reimplement max flow here because we only want to cut outedges
    # When we're not being evaluated.
    scores = np.zeros((N, N))
    for i in xrange(N):
        # Add back in the edges for this agent, so we can get an actual score.
        if i in strategic_agents:
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
        if i in strategic_agents:
            for a, b, _ in saved_edges[i]:
                graph[a][b]['weight'] = THIN_EDGE_WEIGHT

        sys.stdout.write('.')
    sys.stdout.write("\n")

    return scores


def person_shortest_path(graph, num_strategic, sybil_pct):
    """ Personalized Shortest Path.

    No manipulations.
    """
    return tm.shortest_path(graph)

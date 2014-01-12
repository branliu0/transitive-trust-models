""" Strategic Trust Models

This file provides routines that compute various trust model under strategic
manipulation. A given number of agents are chosen to be strategic, and they
perform optimal manipulations to make themselves look better, and then the
trust model scores are applied.
"""
import random

from hitting_time import personalized_LA_ht
import trust_models as tm

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


def global_pagerank(graph, num_strategic, sybil_pct):
    """ Global PageRank.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    strategic_agents = random.sample(graph.nodes(), num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, num_sybils)
    return tm.pagerank(graph)


def person_pagerank(graph, num_strategic, sybil_pct):
    """ Personalized PageRank.

    Cut all outlinks + generate one sybil.
    """
    graph = graph.copy()
    strategic_agents = random.sample(graph.nodes(), num_strategic)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, 1)
    return tm.personalized_pagerank(graph)


def global_hitting_time(graph, num_strategic, sybil_pct):
    """ Global Hitting Time.

    Cut all outlinks + generate sybils.
    """
    graph = graph.copy()
    strategic_agents = random.sample(graph.nodes(), num_strategic)
    num_sybils = int(graph.number_of_nodes() * sybil_pct)
    cut_outlinks(graph, strategic_agents)
    generate_sybils(graph, strategic_agents, num_sybils)
    return tm.hitting_pagerank(graph, 'all')


def person_hitting_time(graph, num_strategic, sybil_pct):
    """ Personalized Hitting Time.

    Cut all outlinks.
    """
    graph = graph.copy()
    strategic_agents = random.sample(graph.nodes(), num_strategic)
    cut_outlinks(graph, strategic_agents)
    return personalized_LA_ht(graph)


def person_max_flow(graph, num_strategic, sybil_pct):
    """ Personalized Max Flow.

    Cut all outlinks.
    """
    graph = graph.copy()
    strategic_agents = random.sample(graph.nodes(), num_strategic)
    cut_outlinks(graph, strategic_agents)
    return tm.max_flow(graph)


def person_shortest_path(graph, num_strategic, sybil_pct):
    """ Personalized Shortest Path.

    No manipulations.
    """
    return tm.shortest_path(graph)

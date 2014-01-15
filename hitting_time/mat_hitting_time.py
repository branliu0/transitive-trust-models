import networkx as nx
import numpy as np

def personalized_eigen_ht(graph, alpha=0.15):
    """
    Makes N^2 eigenvector calculations. Expected O(N^5).
    """
    N = graph.number_of_nodes()
    scores = np.zeros((N, N))
    restarts = np.eye(N)
    old_edges = np.zeros(N)

    # Initially handle dangling nodes and normalization.
    adj_matrix = nx.to_numpy_matrix(graph)
    for k in xrange(N):
        s = adj_matrix[k].sum()
        if s == 0:  # Dangling nodes
            adj_matrix[k, k] = 1
        else:  # Normalization
            adj_matrix[k] /= s

    for i in xrange(N):
        for j in xrange(N):
            # Once you hit j, go straight back to i
            old_edges = adj_matrix[j].copy()
            adj_matrix[j] = restarts[i]

            # Now add in the restart distribution to the matrix.
            htpr_matrix = (1 - alpha) * adj_matrix + \
                    alpha * np.outer(np.ones(N), restarts[i])

            # To obtain PageRank score, take the dominant eigenvector, and
            # pull out the score for node i
            eigenvalues, eigenvectors = np.linalg.eig(htpr_matrix.T)
            dominant_index = eigenvalues.argsort()[-1]
            pagerank = np.array(eigenvectors[:, dominant_index]).flatten().real
            pagerank /= np.sum(pagerank)

            # Using Theorem 1 equation (ii) from Sheldon & Hopcroft 2007 and
            # using the fact that for node i, the expected return time is just
            # one more than the expected hitting time, since the first step
            # away from node i will always be to a node in the pretrusted set,
            # we arrive at this equation for deriving hitting time.
            scores[i][j] = (1.0 / (1 - alpha + alpha / pagerank[j]))

            # And return the adj_matrix to its original state
            adj_matrix[j] = old_edges
    return scores


def personalized_LA_ht(graph, alpha=0.15):
    """ Computes personalized hitting time using a linear equation method.

    This is expected O(N^4).

    This is based on the formula

        (I - (1 - alpha) * M(j)) * h(j) = (1 - alpha) * 1

    which allows us to solve for the h_{ij} for one particular j by solving a
    system of linear equations with N variables and N equations. We repeat this
    N times to obtain all N^2 personalized hitting times.

    Returns:
        An NxN numpy matrix containing the personalized hitting times.
    """
    N = graph.number_of_nodes()
    ht = np.zeros((N, N))
    for i in xrange(N):
        M = nx.to_numpy_matrix(graph)
        for j in xrange(N):
            M[j] /= M[j].sum()
        M[i] = 0
        A = np.eye(N) - (1 - alpha) * M  # The coefficients of the equations
        b = np.repeat(1 - alpha, N)  # The constants of the equations
        ht[:, i] = np.linalg.solve(A, b)  # The solved unknowns
    return -ht  # We negate to reverse the ordering

import networkx as nx
import numpy as np

def personalized_eigen_ht(graph, alpha=0.15):
    """
    Makes N^2 eigenvector calculations.
    """
    N = graph.number_of_nodes()
    scores = np.zeros((N, N))
    restarts = np.eye(N)
    for i in xrange(N):
        for j in xrange(N):
            adj_matrix = nx.to_numpy_matrix(graph)

            # Once you hit j, go straight back to i
            adj_matrix[j] = restarts[i]

            # For dangling nodes, we add a self-edge, which simulates getting
            # "stuck" until we teleport.
            for k in xrange(N):
                if adj_matrix[k].sum() == 0:
                    adj_matrix[k, k] = 1

            # Normalize outgoing edge weights for all nodes.
            for k in xrange(N):
                adj_matrix[k] /= adj_matrix[k].sum()

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
    return scores

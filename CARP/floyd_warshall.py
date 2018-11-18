from numpy import minimum, newaxis


def floyd_warshall_fastest(adjacency_matrix):
    """floyd_warshall_fastest(adjacency_matrix) -> shortest_path_distance_matrix
    Input
        An NxN NumPy array describing the directed distances between N nodes.
        adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
        Notes:
        * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
        * The diagonal of adjacency_matrix should be zero.
    Output
        An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
    """
    n = len(adjacency_matrix)

    for k in range(n):
        adjacency_matrix = minimum(adjacency_matrix, adjacency_matrix[newaxis, k, :] + adjacency_matrix[:, k, newaxis])

    return adjacency_matrix
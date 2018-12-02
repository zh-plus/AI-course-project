from timer import Timer
import numpy as np
from collections import defaultdict


class Graph:
    """
    Implemented by adjacent map
    """

    def __init__(self, V):
        self.V = V
        self.adj = [set() for _ in range(V + 1)]
        self.reverse_adj = [set() for _ in range(V + 1)]
        self.weight = [0 for _ in range(V + 1)]

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.reverse_adj[v].add(u)

    def compute_weight(self):
        out_degrees = [len(s) for s in self.reverse_adj]
        for i, degree in enumerate(out_degrees):
            if degree:
                self.weight[i] = 1 / degree

    def get_children(self, node):
        return self.adj[node]

    def get_parents(self, node):
        return self.reverse_adj[node]

    def get_out_degree(self, node):
        return len(self.adj[node])

    def get_in_degree(self, node):
        return len(self.reverse_adj[node])

    def get_weight(self, inactivate):
        return self.weight[inactivate]


read_line = lambda l: list(map(int, l.split(' ')[:2]))


def graph_from_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        V, E = read_line(lines[0])

        g = Graph(V)
        for line in lines[1:]:
            g.add_edge(*read_line(line))
        g.compute_weight()

        print(g.reverse_adj)
        print(g.weight)

    return g


def seed_from_file(path):
    with open(path, 'r') as f:
        seed = set([int(x) for x in f.readlines()])
    return seed


if __name__ == '__main__':
    path = 'network.txt'
    with Timer():
        g = graph_from_file(path)
    print(g.adj)
    print(g.reverse_adj)
    print(g.get_out_degree(14))
    print(g.get_in_degree(1))

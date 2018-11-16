import numpy as np
from PIL import Image, ImageDraw

from floyd_warshall import floyd_warshall_fastest

get_str_property = lambda s: s.split(': ')[-1].rstrip('\n')
get_int_property = lambda s: int(get_str_property(s))


class Edge:
    def __init__(self, u, v, cost, demand):
        self.u = u
        self.v = v
        self.cost = cost
        self.demand = demand

    def __hash__(self):
        return hash((self.u, self.v)) + hash((self.v, self.u))

    def __eq__(self, other):
        return self.u, self.v == other.u, other.v or self.u, self.v == other.v, other.u

    def __str__(self):
        return str((self.u, self.v, self.cost, self.demand))


class Solution:
    def __init__(self, routes, loads, costs, total_cost, valid=True):
        self.routes = routes
        self.loads = loads
        self.costs = costs
        self.total_cost = total_cost
        self.valid = valid




class CARPInfo:
    """parse the file and construct the graph of this instance
    """

    def __init__(self, instance_path):
        self.instance = open(instance_path, 'r', 51200).readlines()  # 50 KB buffer

        lines = self.instance[:8]
        self.name = get_str_property(lines[0])
        self.vertices = get_int_property(lines[1])
        self.depot = get_int_property(lines[2])
        self.edges_required = get_int_property(lines[3])
        self.edges_nonreq = get_int_property(lines[4])
        self.vehicles = get_int_property(lines[5])
        self.capacity = get_int_property(lines[6])
        self.total_cost = get_int_property(lines[7])

        self.edges = dict()  # all edges
        self.tasks = dict()  # all required edges
        data = map(lambda s: s.split(), self.instance[9:-1])
        arr = np.full((self.vertices + 1, self.vertices + 1), np.inf)  # full of inf, ps: the vertices +1 means start from 1
        np.fill_diagonal(arr, 0)  # make diagonal 0

        for line in data:
            u, v, cost, demand = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            edge = Edge(u, v, cost, demand)
            self.edges[(u, v)] = edge
            self.edges[(v, u)] = edge
            if demand:  # demand != 0
                self.tasks[(u, v)] = edge
                self.tasks[(v, u)] = edge

            arr[u, v] = cost
            arr[v, u] = cost

        self.min_dist = floyd_warshall_fastest(arr)

    def visualise(self, solution):

        im = Image.new('RGB', (500, 500), "white")  # create a new black image
        draw = ImageDraw.Draw(im)
        for i, route in enumerate(solution.routes):
            r_c = (i * i) % 255
            g_c = (i * r_c) % 255
            b_c = (i * g_c) % 255
            nodes = route.route
            norm = lambda x, y: (2 * x + 250, 2 * y + 250)
            draw.line([norm(*self.coords[n]) for n in nodes], fill=(r_c, g_c, b_c), width=2)
        return im

    def __str__(self):
        s = '''name: {}
vertices: {}
depot: {}
edges_required: {}
edges_nonreq: {}
vehicles: {}
capacity: {}
total_cost: {}'''.format(self.name, self.vertices, self.depot, self.edges_required, self.edges_nonreq, self.vehicles, self.capacity, self.total_cost)
        return s

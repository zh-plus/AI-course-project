import numpy as np
# from PIL import Image, ImageDraw

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
    def __init__(self, routes, loads, costs, total_cost, capacity):
        self.routes = routes
        self.loads = loads
        self.costs = costs
        self.total_cost = int(total_cost) if total_cost != np.inf else np.inf

        self.capacity = capacity
        self.load_exceed = sum([c - capacity for c in loads if c > capacity])

        self.is_valid = self.load_exceed == 0
        if self.is_valid:
            self.non_valid_generations = 0
        else:
            self.non_valid_generations = 1

        if self.loads:
            self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.non_valid_generations)

    @staticmethod
    def worst():
        return Solution([], [], [np.inf], np.inf, np.inf)

    def check_valid(self):
        self.load_exceed = sum([c - self.capacity for c in self.loads if c > self.capacity])
        self.is_valid = self.load_exceed == 0
        if not self.is_valid:
            self.non_valid_generations += 1

        self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.non_valid_generations)


        if self.routes.count([]):
            for i, c in enumerate(self.routes):
                if not c:
                    del self.routes[i]
                    del self.loads[i]
                    del self.costs[i]

    @staticmethod
    def generate_from_route(routes, info):
        """

        :type info: CARPInfo
        """
        new_solution = Solution(routes, [], [], 0, info.capacity)
        costs = get_costs(new_solution, info)
        loads = []
        for route in routes:
            loads.append(sum(map(lambda r: info.tasks[(r[0], r[1])].demand, route)))
        new_solution.loads = loads
        new_solution.costs = costs
        new_solution.total_cost = sum(costs)

        new_solution.check_valid()

        return new_solution

    def __hash__(self):
        return hash(str(self.routes))

    def __eq__(self, other):
        return self.routes == other.routes

    def __str__(self):
        return '\n'.join(['routs:' + str(self.routes), 'loads:' + str(self.loads), 'costs:' + str(self.costs), 'total_cost:' + str(self.total_cost),
                          'is_valid:' + str(self.is_valid), 'non_valid_generations:' + str(self.non_valid_generations)])


def get_cost(solution, info):
    """

    :type info: CARPInfo
    """
    cost = 0
    routes = solution.routes
    for route in routes:
        cost += info.min_dist[info.depot, route[0][0]]
        for i in range(len(route)):
            u, v = route[i]
            next_u = route[i + 1][0] if i != len(route) - 1 else info.depot
            cost += info.edges[(u, v)].cost + info.min_dist[v, next_u]
    return cost


def get_costs(solution, info):
    """

    :type info: CARPInfo
    """
    costs = []
    routes = solution.routes
    for route in routes:
        this_cost = info.min_dist[info.depot, route[0][0]]
        for i in range(len(route)):
            u, v = route[i]
            next_u = route[i + 1][0] if i != len(route) - 1 else info.depot
            this_cost += info.edges[(u, v)].cost + info.min_dist[v, next_u]
        costs.append(this_cost)

    return costs


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

    # def visualise(self, solution):
    #     im = Image.new('RGB', (500, 500), "white")  # create a new black image
    #     draw = ImageDraw.Draw(im)
    #     for i, route in enumerate(solution.routes):
    #         r_c = (i * i) % 255
    #         g_c = (i * r_c) % 255
    #         b_c = (i * g_c) % 255
    #         nodes = route.route
    #         norm = lambda x, y: (2 * x + 250, 2 * y + 250)
    #         draw.line([norm(*self.coords[n]) for n in nodes], fill=(r_c, g_c, b_c), width=2)
    #     return im

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

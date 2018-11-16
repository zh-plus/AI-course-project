from typing import Dict, Tuple
import numpy as np
from collections import defaultdict

from CARP_info import CARPInfo, Edge


class CARPAlgorithm:
    def __init__(self, info):
        """

        :type info: CARPInfo
        """
        self.info = info
        self.min_dist = info.min_dist
        self.depot = info.depot
        self.capacity = info.capacity

        from_depot = lambda x: self.min_dist[x, self.depot]
        self.rules = [
            lambda x, y, c: from_depot(x.v) > from_depot(y.v),
            lambda x, y, c: from_depot(x.v) < from_depot(y.v),
            lambda x, y, c: x.demand / x.cost > y.demand / y.cost,
            lambda x, y, c: x.demand / x.cost < y.demand / y.cost,
            lambda x, y, c: from_depot(x.v) > from_depot(y.v) if c < self.capacity / 2 else from_depot(x.v) < from_depot(y.v)
        ]

        self.population = self.get_best_ini()
        # print(self.population)

    def path_scanning(self, rule):
        free: Dict[Tuple[int, int], Edge] = self.info.tasks.copy()
        routes, loads, costs = [], [], []
        while len(free):
            last_end = self.depot
            routes.append([])
            loads.append(0)
            costs.append(0)
            while len(free):
                selected_edge = list(free.values())[0]
                distance = np.inf
                for edge in [x for x in free.values() if loads[-1] + x.demand <= self.capacity]:
                    d = self.min_dist[last_end, edge.u]
                    if d < distance:
                        distance = d
                        selected_edge = edge
                    elif d == distance and self.better(edge, selected_edge, loads[-1], rule):
                        selected_edge = edge

                if distance == np.inf:  # means distance not updated
                    break

                routes[-1].append((selected_edge.u, selected_edge.v))
                free.pop((selected_edge.u, selected_edge.v))
                free.pop((selected_edge.v, selected_edge.u))

                loads[-1] += selected_edge.demand
                costs[-1] += selected_edge.cost + distance  # task_cost and min_dist cost

                last_end = selected_edge.v
            costs[-1] += self.min_dist[last_end, self.depot]

        # print('routes:', routes)
        # print('loads:', loads)
        # print('costs:', costs)
        # print('sum_cost:', sum(costs))
        result = {
            'routes': routes,
            'loads': loads,
            'costs': costs,
            'total_cost': int(sum(costs))
        }

        return result

    def get_best_ini(self):
        best_result = defaultdict(lambda: np.inf)
        for rule in self.rules:
            result = self.path_scanning(rule)
            total_cost = result['total_cost']
            if total_cost < best_result['total_cost']:
                best_result = result

        return best_result

    def step(self):
        return self.population

    def better(self, edge, selected_task, current_load, rule):
        return rule(edge, selected_task, current_load)

import random
import numpy as np
from time import perf_counter
from copy import copy
from argparse import ArgumentParser

from graph import Graph, graph_from_file, seed_from_file


def rand_happen(p):
    return True if random.random() <= p else False


class ISE:
    def __init__(self, graph=None, seeds=None, model=None, time_limit=None, step_num=5000):
        """

        :type seeds: set
        :type graph: Graph
        """
        if not graph:
            args = self.input_parser()
        self.graph = graph_from_file(args.i) if not graph else graph
        self.seeds = seed_from_file(args.s) if not seeds else seeds
        self.model = args.m if not model else model
        self.time_limit = args.t if not graph else time_limit
        self.step_num = step_num

    @staticmethod
    def input_parser():
        parser = ArgumentParser()
        parser.add_argument('-i', type=str)
        parser.add_argument('-s', type=str)
        parser.add_argument('-m', type=str, choices=('IC', 'LT'))
        parser.add_argument('-t', type=int)

        return parser.parse_args()

    # def run_time_limit(self):  # according to time limit
    #     start = perf_counter()
    #
    #     step = self.LT if self.model == 'LT' else self.IC
    #
    #     # pool = Pool(processes=8)
    #
    #     self.time_limit -= perf_counter() - start
    #
    #     influence = []
    #     avg_time = 0
    #     counter = 0
    #     while self.time_limit > avg_time * 1.5:
    #         this_start = perf_counter()
    #
    #         influence.append(step())
    #
    #         this_time = perf_counter() - this_start
    #         avg_time = 0.2 * avg_time + this_time * 0.8
    #
    #         # print('this_time: {}\tavg_time: {}\ttime_left: {}'.format(this_time, avg_time, self.time_limit))
    #         self.time_limit -= this_time
    #         counter += 1
    #
    #     print('counter:', counter)
    #
    #     return np.average(influence)

    def run_step_limit(self):  # according to step limit
        step = self.LT if self.model == 'LT' else self.IC

        influence = [step() for _ in range(self.step_num)]

        return np.average(influence)

    def IC(self):
        graph = self.graph
        active_set = copy(self.seeds)
        all_active_set = copy(self.seeds)

        influenced = len(active_set)
        while len(active_set):
            new_active_set = set()
            for active_node in active_set:
                inactive_neighbours = graph.get_children(active_node) - all_active_set
                for inactive_node in inactive_neighbours:
                    weight = graph.get_weight(inactive_node)
                    if rand_happen(weight):
                        all_active_set.add(inactive_node)
                        new_active_set.add(inactive_node)

            influenced += len(new_active_set)
            active_set = new_active_set

        return influenced

    def LT(self):
        graph = self.graph
        active_set = copy(self.seeds)
        pre_active_set = copy(self.seeds)

        threshold = np.random.random(graph.node_size + 1)
        influenced = len(active_set)
        while len(active_set):
            new_active_set = set()
            for active_node in active_set:
                inactive_neighbours = graph.get_children(active_node) - pre_active_set
                for inactive_node in inactive_neighbours:
                    active_neighbours = graph.get_parents(inactive_node) & pre_active_set
                    total_weight = graph.get_weight(inactive_node) * len(active_neighbours)
                    if total_weight >= threshold[inactive_node]:
                        pre_active_set.add(inactive_node)
                        new_active_set.add(inactive_node)

            influenced += len(new_active_set)
            active_set = new_active_set

        return influenced


if __name__ == '__main__':
    import sys

    start = perf_counter()

    if len(sys.argv) == 1:
        sys.argv = ['ISE.py', '-i', 'network.txt', '-s', 'seeds2.txt', '-m', 'IC', '-t', '1']

    ise = ISE()
    elapse = perf_counter() - start
    ise.time_limit -= elapse

    result = ise.run_step_limit()
    print(result)
    print(perf_counter() - start, 's')
    print(elapse)

    sys.exit(0)

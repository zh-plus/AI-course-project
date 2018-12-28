import numpy as np
from argparse import ArgumentParser
from time import perf_counter
# from queue import PriorityQueue
from multiprocessing import Pool

from graph import Graph, graph_from_file, compute_n
from ISE import ISE


class IMP:
    def __init__(self, graph=None, seed_size=None, model=None, time_limit=None):
        """

        :type graph: Graph
        """
        start = perf_counter()

        args = self.input_parser()
        self.graph = graph_from_file(args.i) if not graph else graph
        self.seed_size = args.k if not seed_size else seed_size
        self.model = args.m if not model else model
        self.time_remain = args.t if not time_limit else time_limit
        self.seeds = set()
        self.last_ise = 0

        self.algorithms = {
            'CELF': self.celf
        }

        self.compute_n = compute_n(self.graph.node_size, self.graph.edge_size)
        # print(self.compute_n)

        self.time_remain -= perf_counter() - start

    @staticmethod
    def input_parser():
        parser = ArgumentParser()
        parser.add_argument('-i', type=str)
        parser.add_argument('-k', type=int)
        parser.add_argument('-m', type=str, choices=('IC', 'LT'))
        parser.add_argument('-t', type=int)

        return parser.parse_args()

    def run(self, alg):
        return self.algorithms[alg]()

    def celf(self):
        graph = self.graph

        start = perf_counter()

        # candidates = [[i, self.compute_diff(i)] for i in range(1, graph.node_size + 1)]  # \\TODO priorityqueue approach

        # candidates = []  # \\TODO priorityqueue approach
        # for i in range(1, graph.node_size + 1):
        #     candidates.append([i, self.compute_diff(i)])

        with Pool(8) as p:
            candidates_value = p.map(self.compute_diff, range(1, graph.node_size + 1))
        candidates = [[i, candidates_value[i - 1]] for i in range(1, graph.node_size + 1)]

        end = perf_counter()
        self.time_remain -= (end - start)
        # print(end - start, 's for first iteration')

        avg_time = 0
        seeds = self.seeds
        while len(seeds) < self.seed_size and self.time_remain > 4 * avg_time + 2:
            iter_start = perf_counter()

            computed = set()  # \\TODO list approach should be tested
            while 1:
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                node, diff = candidates[0]
                if node in computed:
                    seeds.add(node)
                    # print('1:', node)
                    self.last_ise += diff
                    del candidates[0]
                    break
                else:
                    candidates[0][1] = self.compute_diff(node)
                    computed.add(node)

            iter_time = perf_counter() - iter_start
            self.time_remain -= iter_time
            avg_time = 0.1 * avg_time + 0.9 * iter_time
            # print(avg_time, self.time_remain)

        if len(seeds) < self.seed_size:  # if not completed, using left candidates //TODO degree_discount
            less = self.seed_size - len(seeds)
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[0:less]
            for x in candidates:
                seeds.add(x[0])

        return seeds

    def compute_diff(self, s):
        if not self.graph.get_out_degree(s):
            return 0

        ise = ISE(self.graph, self.seeds | {s}, self.model, step_num=self.compute_n)
        # if s % 100 == 0:
        #     print(s * 100 / self.graph.node_size, '%')

        return ise.run_step_limit() - self.last_ise


if __name__ == '__main__':
    import sys

    # start = perf_counter()

    if len(sys.argv) == 1:
        sys.argv = ['IMP.py', '-i', 'network.txt', '-k', '5', '-m', 'LT', '-t', '60']

    imp = IMP()
    # elapse = perf_counter() - start
    # imp.time_remain -= elapse

    result = imp.run('CELF')
    for x in result:
        print(x)
    # print(imp.last_ise)
    #
    # print(perf_counter() - start, 's')

    sys.exit(0)

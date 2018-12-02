from argparse import ArgumentParser
from time import perf_counter

from graph import Graph, graph_from_file


class IMP:
    def __init__(self, graph=None, seed_size=None, model=None, time_limit=None):
        """

        :type seeds: list
        :type graph: Graph
        """
        args = self.input_parser()
        self.graph = graph_from_file(args.i) if not graph else graph
        self.seed_size = args.k if not seed_size else seed_size
        self.model = args.m if not model else model
        self.time_limit = args.t if not time_limit else time_limit

    @staticmethod
    def input_parser():
        parser = ArgumentParser()
        parser.add_argument('-i', type=str)
        parser.add_argument('-k', type=int)
        parser.add_argument('-m', type=str, choices=('IC', 'LT'))
        parser.add_argument('-t', type=int)

        return parser.parse_args()


if __name__ == '__main__':
    import sys

    start = perf_counter()

    if len(sys.argv) == 1:
        sys.argv = ['ISE.py', '-i', 'network.txt', '-s', 'seeds2.txt', '-m', 'IC', '-t', '5']

    imp = IMP()
    elapse = perf_counter() - start
    imp.time_limit -= elapse

    result = imp.run()
    print(result)

    print(perf_counter() - start, 's')

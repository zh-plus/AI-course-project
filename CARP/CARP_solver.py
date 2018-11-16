import numpy as np
from time import time
import random

from CARP_algorithm import CARPAlgorithm
from CARP_info import CARPInfo


class CARPHandler:
    def __init__(self, instance_path, termination, seed, iterations=60):
        self.info = CARPInfo(instance_path)
        self.termination = termination
        self.iterations = iterations

        # print(self.info)

        random.seed(seed)

    def handle_output(self, solution):
        def s_format(s):
            s_print = []
            for p in s:
                s_print.append(0)
                s_print.extend(p)
                s_print.append(0)
            return s_print

        routes = solution['routes']
        print("s", (",".join(str(d) for d in s_format(routes))).replace(" ", ""))
        print("q", solution['total_cost'])

    def run(self):
        solver = CARPAlgorithm(self.info)
        start_time = time()

        iter_num = 0
        best = None  # result dict
        while iter_num < self.iterations:
            # print('iteration {}'.format(iter_num))
            best = solver.step()
            iter_num += 1

            if time() - start_time > self.termination:
                break
        self.handle_output(best)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv = ['CARP_solver.py', 'C:\\Users\\10578\\PycharmProjects\\AICourse\\CARP\\CARP_samples\\egl-s1-A.dat', '-t', '600', '-s', '1']

    path, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    termination, seed = int(termination), int(seed)
    # print(file, termination, seed)
    handler = CARPHandler(path, int(termination), seed, iterations=1)
    handler.run()

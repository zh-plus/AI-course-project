import numpy as np
from time import perf_counter
import random
import os

from CARP_algorithm import CARPAlgorithm
from CARP_info import CARPInfo, get_cost


class CARPHandler:
    def __init__(self, instance_path, termination, seed, population_size=100, test=False):
        self.info = CARPInfo(instance_path)
        self.termination = termination
        self.test = test
        self.path = instance_path
        self.population_size = population_size

        # print(self.info)

        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def handle_output(solution):
        def s_format(s):
            s_print = []
            for p in s:
                s_print.append(0)
                s_print.extend(p)
                s_print.append(0)
            return s_print

        routes = solution.routes
        print("s", (",".join(str(d) for d in s_format(routes))).replace(" ", ""))
        print("q", solution.total_cost)

    def run(self):
        avg_time = 0
        total_time = 0
        time_remain = self.termination - 2
        start_time = perf_counter()

        solver = CARPAlgorithm(self.info, self.population_size)
        time_remain -= perf_counter() - start_time

        iter_num = 0
        best = None  # result dict
        while time_remain > 2 * avg_time:
            iter_start = perf_counter()
            best = solver.step()
            iter_end = perf_counter()

            iter_time = iter_end - iter_start
            iter_num += 1
            total_time += iter_time
            avg_time = 0.6 * avg_time + 0.4 * iter_time
            time_remain -= iter_time
            if self.test:
                print('pid: {} \tsample: {} \titer {} \t\tpopulation: {} \ttime: {:5.3} s \tavg: {:5.3} s \tremain: {:.3f} s \tcost: {}'.format(
                    os.getpid(),
                    self.path.split('\\')[-1],
                    iter_num,
                    len(solver.population),
                    iter_time,
                    avg_time,
                    time_remain,
                    best.total_cost)
                )

        # self.handle_output(best)
        return int(best.total_cost)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv = ['CARP_solver.py', 'CARP_samples\egl-s1-A.dat', '-t', '10', '-s', '1']

    path, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    termination, seed = int(termination), int(seed)
    # print(file, termination, seed)
    handler = CARPHandler(path, int(termination), seed, test=True)
    handler.run()
    sys.exit(0)

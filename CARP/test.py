import os
import time


class Timer(object):
    def __init__(self):
        self.start_time = time.perf_counter()

    def stop(self):
        print('{:.3f} seconds elapsed'.format(time.perf_counter() - self.start_time))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


with Timer():
    os.system('python CARP_solver.py C:\\Users\\10578\\PycharmProjects\\AICourse\\CARP\\CARP_samples\\gdb1.dat -t 10 -s 1')
    # print(1)
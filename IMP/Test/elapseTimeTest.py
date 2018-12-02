from timer import Timer
from time import perf_counter


this_start = 0.0000001
with Timer():
    for _ in range(465504):
        this_time = perf_counter() - this_start
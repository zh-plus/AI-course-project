import time
import sys

t = time.perf_counter()
time.sleep(2)
print(time.perf_counter() - t, 's')

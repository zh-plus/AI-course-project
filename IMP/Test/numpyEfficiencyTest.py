from timer import Timer

import numpy as np

# append
with Timer():
    list1 = []
    for x in range(100000):
        list1.append(x)
# read
with Timer():
    for x in range(100000):
        y = list1[x]

# append
with Timer():
    list2 = np.array([])
    for x in range(100000):
        list2 = np.append(list2, x)
# read
with Timer():
    for x in range(100000):
        y = list2[x]

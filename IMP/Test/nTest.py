import numpy as np
from math import log


E1, E2 = 159, 58891
V1, V2 = 62, 15233

u = (E1 * V1)
v = (E2 * V2)

p = v / u
# q = log(200, p)
q = 0.5
print('q:', q)

x1 = 1 / (u ** q)
x2 = 1 / (v ** q)
print('u ^ q:', x1)
print('v ^ q:', x2)
x = np.array([[x1, x2]])
x = np.row_stack((x, [1, 1]))
print(x)

n = [10000, 500]

w = np.dot(n, np.linalg.inv(x))
print(w)
a, b = w


def compute_n(V, E, q):
    global a, b
    return 946368.048 / ((V * E) ** 0.5) + 468.403216


print(compute_n(15233, 58891, q))
import re

import numpy as np

state: np.ndarray = np.array([
    [ 1, 1, 1,  1],
    [-1, 1, 0,  1],
    [ 1, 0, 1,  1],
    [-1, 1, 1, -1]
])
np.where(state == -1)

to_str = lambda array: ''.join(map(str, array))


# re_counter = lambda pattern, string: len(re.findall(pattern, string))

def re_counter(pattern, string):
    result = re.findall(pattern, string)

    if len(result) != 0:
        print('{:>30}  :  {}'.format(pattern, string))

    return len(result)


pattern_value = {
    '[^-]?(1){4}': 100000,      # 连5
    '[^-1]?(1){3}[^(-1)]?': 10000,      # 连3 #TODO
}

mark = 0
for i in range(len(state)):
    horizontal = to_str(state[i])
    vertical = to_str(state[..., i])
    lr_diag = to_str(state.diagonal(i))
    rl_diag = to_str(state.diagonal(-i))


    # def counter(pattern):
    #     a = re_counter(pattern, horizontal)
    #     b = re_counter(pattern, vertical)
    #     c = re_counter(pattern, lr_diag)
    #     d = re_counter(pattern, rl_diag)
    #     return pattern_value[pattern] * (a + b + c + d)


    counter = lambda pattern: pattern_value[pattern] * (re_counter(pattern, horizontal) + re_counter(pattern, vertical)
                                                        + re_counter(pattern, lr_diag) + re_counter(pattern, rl_diag))

    mark += sum(map(counter, pattern_value.keys()))

    # for pattern, value in pattern_value.items():
    #     mark += counter(pattern)

print(mark)

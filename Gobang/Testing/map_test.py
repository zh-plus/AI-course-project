l = [1, 2, 3]
d = {1: 4, 2: 5, 3: 6}

res = sum(map(lambda x: d[x], l))
print(res)
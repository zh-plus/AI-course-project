d = {1: '1', 2: '123'}
d1 = d.copy()
d1.pop(1)
d1[2] = '12'
print(d)
print(d1)
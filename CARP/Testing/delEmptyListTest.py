l = [[1, 3, 4], [], [1], []]
for i, li in enumerate(l):
    if not li:
        del l[i]
print(l)
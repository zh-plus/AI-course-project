read_line = lambda l: list(map(int, l.split(' ')[:2]))


def graph_from_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        ns = []
        for line in lines[1:]:
            u, v = read_line(line)
            ns.append(u)
            ns.append(v)

    print(max(ns))


graph_from_file('Amazon.txt')

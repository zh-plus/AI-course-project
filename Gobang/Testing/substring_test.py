from time import time

a = '101011011311012101'

test = False

tic = time()

for x in range(1000000):
    # test = a.__contains__('311011')
    # test = '311011' in a
    # test = a.find('311011')
    test = '311011' in str.che

print(time() - tic, 's')
from time import time


string = '001121021111010'
substr = '211110'

test_range = range(1000000)
result = None

tic = time()
for x in test_range:
    result = substr in string
print('in:', time() - tic, 's')

tic = time()
for x in test_range:
    result = string.__contains__(substr)
print('contains:', time() - tic, 's')

tic = time()
for x in test_range:
    result = string.find(substr)
print('find:', time() - tic, 's')

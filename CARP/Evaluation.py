from CARP_solver import CARPHandler
import os
import time


def evaluate(data):
    cost = CARPHandler(data, 120, 1, test=True).run()
    result = (data.split('\\')[-1], cost)
    return result
    # time.sleep(1)
    # print('pid: {} \tdata: {}'.format(os.getpid(), data))


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, abspath
    from multiprocessing import Pool

    data_path = 'CARP_samples/eglese'
    test_data = [abspath(join(data_path, f)) for f in listdir(data_path) if isfile(join(data_path, f))]
    print(test_data)

    pool = Pool(processes=8)
    result = []
    for i in range(len(test_data)):
        r = pool.apply_async(evaluate, (test_data[i], ))
        result.append(r)

    pool.close()
    pool.join()

    print(result)

    result_file = open('result.txt', 'w')
    result_file.writelines(['data: {} \t result: {}\n'.format(r.get()[0], r.get()[1]) for r in result])

    # if len(sys.argv) == 1:
    #     sys.argv = ['CARP_solver.py', 'E:\Python\AICourse\CARP\CARP_samples\egl-e1-A.dat', '-t', '120', '-s', '1']
    #
    # path, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    # termination, seed = int(termination), int(seed)
    # # print(file, termination, seed)
    # handler = CARPHandler(path, int(termination), seed, test=True)
    # handler.run()
    # sys.exit(0)

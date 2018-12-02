from CARP_solver import CARPHandler
import os
import time
import xlwt
from time import perf_counter


def evaluate(data, population_size):
    cost = CARPHandler(data, 3, 1, population_size=population_size, test=False).run()
    result = (data.split('\\')[-1], cost)
    return result
    # time.sleep(1)
    # print('pid: {} \tdata: {}'.format(os.getpid(), data))


def write_to_excel(result):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('size tuning')
    sheet.write(0, 1, 'eglese')
    sheet.write(0, 2, 'bccm')
    sheet.write(0, 3, 'gdb')
    for x in range(len(result[1])):
        sheet.write(x + 1, 0, x * 5 + 10)

    for i in range(len(result)):
        for j in range(len(result[i])):
            sheet.write(j + 1, i + 1, result[i][j][1])
    wbk.save('result.xls')


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, abspath
    from multiprocessing import Pool

    start_time = perf_counter()

    data_paths = ['CARP_samples/eglese', 'CARP_samples/bccm', 'CARP_samples/gdb']
    # data_paths = ['CARP_samples/eglese', 'CARP_samples/bccm']
    p_sizes = range(40, 171, 5)

    size_result = [[], [], []]
    for i, data_path in enumerate(data_paths):
        for population_size in p_sizes:
            test_data = [abspath(join(data_path, f)) for f in listdir(data_path) if isfile(join(data_path, f))]

            pool = Pool(processes=16)
            result = []
            for _ in range(20):
                for j in range(len(test_data)):
                    r = pool.apply_async(evaluate, (test_data[j], population_size,))
                    result.append(r)

            pool.close()
            pool.join()

            size_result[i].append((population_size, sum(map(lambda x: x.get()[1], result)) / len(result)))

            print('data: {} \tsize: {} \tdone\n'.format(data_path, population_size))

        result_file = open('result_{}.txt'.format(data_path.split('/')[1]), 'w')

        result_file.writelines(['size: {} \tavg_cost: {} \n'.format(r[0], r[1]) for r in size_result[i]])

    print(size_result)
    print('time consumed: {} s'.format(perf_counter() - start_time))
    write_to_excel(size_result)

    # if len(sys.argv) == 1:
    #     sys.argv = ['CARP_solver.py', 'E:\Python\AICourse\CARP\CARP_samples\egl-e1-A.dat', '-t', '120', '-s', '1']
    #
    # path, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    # termination, seed = int(termination), int(seed)
    # # print(file, termination, seed)
    # handler = CARPHandler(path, int(termination), seed, test=True)
    # handler.run()
    # sys.exit(0)

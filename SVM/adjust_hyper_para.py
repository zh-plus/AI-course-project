from multiprocessing import Pool
from pprint import pprint
from SVM import SVM

import matplotlib.pyplot as plt
import numpy as np
import os
import time


def train_and_test(C, tolerance):
    svm = SVM(C=C, tolerance=tolerance)
    svm.train()
    acc = svm.test()
    # print(os.getpid(), acc)
    return [acc, svm]


def atomic(C, tolerance):
    with Pool(16) as p:
        future_results = [p.apply_async(train_and_test, args=(C, tolerance)) for i in range(16)]
        results = np.array([f.get() for f in future_results])
    # svm = SVM()
    # pprint(results)
    max_acc = max(results, key=lambda x: x[0])
    # print(max(results, key=lambda x: x[0]))

    return max_acc[0]


def plot(result):
    x, y = zip(*result)
    plt.scatter(x, y)
    # plt.title('')
    plt.xlabel('parameters')
    plt.ylabel('accuracy')

    this_time = time.strftime('%d_%H_%M_%S', time.localtime(time.time()))
    plt.savefig('parameter_' + this_time)


def optimize_C(start=8, end=13, step=1, tolerance=0.001):
    Cs = range(start, end, step)
    results = []
    for C in Cs:
        result = []
        for i in range(10):
            result.append(atomic(C, tolerance))
            print(i, end=' ')
        print()

        mean = np.mean(result)
        print(f'{C}: {mean}')
        results.append([C, mean])

    plot(results)


def optimize_t(C=9, tolerance=0.001):
    ts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    results = []
    for t in ts:
        result = []
        for i in range(10):
            result.append(atomic(C, t))
            print(i, end=' ')
        print()

        mean = np.mean(result)
        print(f'{t}: {mean}')
        results.append([t, mean])

    plot(results)


if __name__ == '__main__':
    optimize_t()

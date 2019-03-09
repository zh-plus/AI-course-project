from pprint import pprint
import random
import numpy as np

# random.seed(1)


def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
        data = list(map(lambda s: s.split(' '), data))
        for d in data:
            d[-1] = d[-1].strip()

        for i in range(len(data)):
            data[i] = list(map(float, data[i]))

        return np.array(data)


def split_data(data):
    random.shuffle(data)
    # training: testing = 7: 3
    split_point = (len(data) * 9) // 10
    return data[: split_point], data[split_point:]


def convert_format_to_their():
    file = open('my_data.txt', 'w')
    with open('train_data.txt', 'r') as f:
        data = f.readlines()
        data = list(map(lambda s: s.replace(' ', ','), data))
        file.writelines(data)


def convert_format_to_me():
    file = open('my_data_cancer.txt', 'w')
    with open('breast_cancer_train.txt', 'r') as f:
        data = f.readlines()
        data = list(map(lambda s: s.replace('\t', ' '), data))
        file.writelines(data)


if __name__ == '__main__':
    convert_format_to_me()

from pprint import pprint
import random

random.seed(1)


def read_data():
    with open('train_data.txt', 'r') as f:
        data = f.readlines()
        data = list(map(lambda s: s.split(' '), data))
        for d in data:
            d[-1] = d[-1].strip()

        return data


def split_data(data):
    random.shuffle(data)
    # training: testing = 7: 3
    split_point = (len(data) * 7) // 10
    return data[: split_point], data[split_point:]


if __name__ == '__main__':
    data = read_data()
    training_data, testing_data = split_data(data)
    print(training_data)
    print(testing_data)
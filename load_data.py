import os


def load(datafile):
    dataset = []
    with open(datafile, 'r', encoding='UTF-8') as data:
        for line in data.readlines():
            line = line.strip().split(',')
            dataset.append([word for word in line[1].split(' ')])
    return dataset


if __name__ == '__main__':
    datafile = "data/data.txt"
    load(datafile)

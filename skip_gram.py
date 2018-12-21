from load_data import load
import collections


def read_data():
    datafile = "data/data.txt"
    dataset = load(datafile)
    words = []
    for num in range(len(dataset)):
        for word in dataset[num]:
            words.append(word)
    return words


def build_dataset(words, min_count):
    count = [['UNK', -1]]
    count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # dictionary count,
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, window_size, data):
    pass


def word2vec_model():
    pass


def save_embedding():
    pass


if __name__ == '__main__':
    words = read_data()
    data, count, dictionary, reverse_dictionary = build_dataset(words, 5)









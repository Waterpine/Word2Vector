from load_data import load
import collections
import numpy as np
import time
import random
import tensorflow as tf


def read_data():
    datafile = "data/data.txt"
    dataset = load(datafile)
    words = []
    for num in range(len(dataset)):
        for word in dataset[num]:
            words.append(word)
    return words


def one_hot_embed(num, vocabulary_size):
    emb_line = np.zeros(vocabulary_size)
    emb_line[num] = 1
    return emb_line


def build_dataset(words, min_count):
    """
    :param words: word list
    :param min_count: threshold
    :return: words_num: char to num
    :return: count: every word appearance times
    :return: dictionary:
    :return: reverse_dictionary:
    """
    count = [['UNK', -1]]
    count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    words_num = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        words_num.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # dictionary count,
    return words_num, count, dictionary, reverse_dictionary


# N-gram
def get_targets(words_num, idx, window_size=5):
    """
    :param words_num:
    :param idx:
    :param window_size:
    :return:
    """
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words_num[start_point: idx] + words_num[idx+1: end_point+1])
    return list(targets)


def get_batch(words_num, batch_size, vocabulary_size, window_size=5):
    """
    :param words_num:
    :param batch_size:
    :param window_size:
    :return:
    """
    n_batch = len(words_num) // batch_size
    words_num = words_num[: n_batch * batch_size]
    for idx in range(0, len(words_num), batch_size):
        x, y = [], []
        batch = words_num[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            x_emb = one_hot_embed(batch_x, vocabulary_size=vocabulary_size)
            for _ in range(len(batch_y)):
                x.append(x_emb)
            # x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


def word2vec_model(vocabulary_size, batch_size, embedding_size, num_sampled, words_num):
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, vocabulary_size], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))
        embed = tf.matmul(inputs, embeddings)
        # embed = tf.nn.embedding_lookup(embeddings, inputs)
        weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.1), dtype=tf.float32)
        biases = tf.Variable(tf.zeros(vocabulary_size), dtype=tf.float32)
        loss = tf.nn.nce_loss(weights=weights,
                              biases=biases,
                              labels=labels,
                              inputs=embed,
                              num_sampled=num_sampled,
                              num_classes=vocabulary_size)
        # loss = tf.nn.sampled_softmax_loss(weights, biases, labels, embed, num_sampled, vocabulary_size)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        epochs = 10
        window_size = 5
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())
        for e in range(1, epochs + 1):
            batches = get_batch(words_num, batch_size, vocabulary_size, window_size)
            start = time.time()
            for x, y in batches:
                # print("x:", x)
                # print("y:", y)
                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                loss += train_loss
                if iteration % 10 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 10),
                          "{:.4f} sec/batch".format((end - start) / 10))
                    loss = 0
                    start = time.time()

                iteration += 1


if __name__ == '__main__':
    words = read_data()
    words_num, count, dictionary, reverse_dictionary = build_dataset(words, 5)
    word2vec_model(len(dictionary), 128, 150, 100, words_num)



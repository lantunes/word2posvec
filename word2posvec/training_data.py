try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np


class TrainingData:
    def __init__(self, data, tag_to_index, index_to_tag, word_to_index, index_to_word):
        self.data = data
        self.tag_to_index = tag_to_index
        self.index_to_tag = index_to_tag
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def to_one_hot(self):
        data = np.array(self.data)
        word_indices = data[:, 0]
        tag_indices = data[:, 1]
        return self._one_hot(word_indices, len(self.word_to_index)), self._one_hot(tag_indices, len(self.tag_to_index))

    @staticmethod
    def _one_hot(x, k, dtype=np.float32):
        """Create a one-hot encoding of x of size k."""
        return np.array(x[:, None] == np.arange(k), dtype)

    @staticmethod
    def from_tagged_words(tagged_words):

        # create the tag indices
        tags = set()
        tag_to_index = {}
        index_to_tag = {}
        tag_index = 0
        for w in tagged_words:
            tag = w[1]
            if tag in tags:
                continue
            tags.add(tag)
            tag_to_index[tag] = tag_index
            index_to_tag[tag_index] = tag
            tag_index += 1

        # create the word indices
        words = set()
        word_to_index = {}
        index_to_word = {}
        word_index = 0
        for w in tagged_words:
            word = w[0]
            if word in words:
                continue
            words.add(word)
            word_to_index[word] = word_index
            index_to_word[word_index] = word
            word_index += 1

        # create the training data, one data point per non-stop word
        data = []
        for w in tagged_words:
            word = w[0]
            data.append((word_to_index[word], tag_to_index[w[1]]))

        return TrainingData(data, tag_to_index, index_to_tag, word_to_index, index_to_word)

    @staticmethod
    def save(training_data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
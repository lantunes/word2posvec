from nltk.corpus import brown
from nltk.corpus import stopwords
from word2posvec import TrainingData
import string

import nltk

if __name__ == '__main__':

    btw = brown.tagged_words(tagset='universal')

    stop = stopwords.words("english") + list(string.punctuation) + ["``", "''", "--"]

    tagged_words = [(w[0].lower(), w[1]) for w in btw if w[0].lower() not in stop and w[0].lower().isalpha()]

    d = nltk.FreqDist(w[0] for w in tagged_words)
    most_common = [c[0] for c in d.most_common(5000)]

    tagged_words = [w for w in tagged_words if w[0] in most_common]

    print(tagged_words)

    training_data = TrainingData.from_tagged_words(tagged_words)

    TrainingData.save(training_data, "brown.top5k.training.data")

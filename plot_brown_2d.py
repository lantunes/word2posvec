import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from word2posvec import Trainer, TrainingData

if __name__ == '__main__':
    td = TrainingData.load("examples/brown.top5k.training.data")
    embeddings = Trainer.load_embeddings("examples/brown.top5k.dim2.embeddings")

    # NOTE: sometimes 'walk' is associated with a different part of speech, like 'NOUN', for example
    words = [
        ('hear', 'VERB'), ('said', 'VERB'), ('run', 'VERB'), ('walk', 'VERB'), ('choose', 'VERB'), ('cook', 'VERB'),
        ('finish', 'VERB'), ('listen', 'VERB'), ('told', 'VERB'), ('know', 'VERB'),

        ('writers', 'NOUN'), ('man', 'NOUN'), ('woman', 'NOUN'), ('person', 'NOUN'), ('step', 'NOUN'),
        ('church', 'NOUN'), ('general', 'NOUN'), ('work', 'NOUN'),

        ('one', 'NUM'), ('five', 'NUM'), ('thousand', 'NUM'), ('forty', 'NUM'),

        ('technical', 'ADJ'), ('conscious', 'ADJ'), ('strange', 'ADJ'), ('aware', 'ADJ'), ('late', 'ADJ'),

        ('even', 'ADV'), ('almost', 'ADV'),

        ('every', 'DET'), ('another', 'DET'), ('either', 'DET')
    ]

    word_indices = []
    colors = []
    for w in words:
        word_index = td.word_to_index[w[0]]
        word_indices.append(word_index)
        pos = w[1]
        if pos == "VERB":
            colors.append("red")
        elif pos == "NOUN":
            colors.append("blue")
        elif pos == "NUM":
            colors.append("black")
        elif pos == "ADJ":
            colors.append("green")
        elif pos == "ADV":
            colors.append("yellow")
        elif pos == "DET":
            colors.append("brown")
        else:
            colors.append("gray")

    X = np.array([embeddings[i] for i in word_indices])

    x, y = X[:, 0], X[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=colors)

    texts = []
    for i, w in enumerate(word_indices):
        texts.append(plt.text(x[i], y[i], td.index_to_word[w]))

    adjust_text(texts, text_from_text=False)

    plt.show()

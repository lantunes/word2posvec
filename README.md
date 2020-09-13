# word2posvec

A Word can have different meanings, depending on the context. Similarly,
words can have different parts of speech, depending also on the context.
Approaches such as word2vec and GloVe produce distributed word
representations according to their semantic content, as given by their
proximity to other words. The approach described in this project
produces distributed word representations according to their parts of
speech. Such representations are useful for tasks such as text
classification, where such encodings attach more grammatical information
to a word.

For example, take the word "work". In any given corpus, the word may be
used as a verb or a noun. Depending on the corpus, it may be used more
as a noun, or more as a verb. A distributed word representation that
captures the "noun-ness" and "verb-ness" of the word would be useful in
a task such as sentence disambiguation, where it is helpful to know how
a word might typically be used.

Word embeddings learned by word2vec or GloVe tend to cluster together
according to their meaning. Word embeddings learned by word2posvec tend
to cluster together according to their part of speech. A word2posvec
embedding for the word "work" would lie somewhere between the nouns and
the verbs (depending on the corpus).

![Results](resources/brown.top5k.dim2.png)

Plot of learned word2posvec embeddings with 2 components. Red points are
verbs, blue points are nouns, black points are numbers, green points are
adjectives, and yellow points are adverbs.

## word2vec

<img src="resources/word2vec.png" alt="word2vec" width="310"/>

A corpus consists of a sequence of <img src="https://render.githubusercontent.com/render/math?math=T"/> words
<img src="https://render.githubusercontent.com/render/math?math=w_{1},%20w_{2},...,%20w_{T}"/>,
and with a context window of size <img src="https://render.githubusercontent.com/render/math?math=c"/>,
the task is to maximize the average log probability:

<img src="resources/eqn1.png" alt="\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \leqslant j \leqslant c, j \neq 0}log \ p(w_{t+j}|w_{t})" width="250"/>

## word2posvec

<img src="resources/word2posvec.png" alt="word2posvec" width="310"/>

A tagged corpus consists of a sequence of <img src="https://render.githubusercontent.com/render/math?math=T"/>
words <img src="https://render.githubusercontent.com/render/math?math=w_{1},%20w_{2},...,%20w_{T}"/>,
and a sequence of corresponding <img src="https://render.githubusercontent.com/render/math?math=T"/>
part of speech tags <img src="https://render.githubusercontent.com/render/math?math=s_{1},%20s_{2},...,%20s_{T}"/>.
The task is to maximize the average log probability:

<img src="resources/eqn2.png" alt="\frac{1}{T}\sum_{t=1}^{T}log \ p(s_{t}|w_{t})" width="150"/>

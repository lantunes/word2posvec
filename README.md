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

A corpus consists of a sequence of words <img src="https://render.githubusercontent.com/render/math?math=w_{1},%20w_{2},...,%20w_{T}"/>,
and with a context window of size <img src="https://render.githubusercontent.com/render/math?math=c"/>,
the task is to maximize the average log probability:

![\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \leqslant j \leqslant c, j \neq 0}log \ p(w_{t+j}|w_{t})](resources/eqn1.png)

## word2posvec

<img src="resources/word2posvec.png" alt="word2posvec" width="310"/>

A tagged corpus consists of words <img src="https://render.githubusercontent.com/render/math?math=W"/>,
and each word <img src="https://render.githubusercontent.com/render/math?math=w%20\in%20W"/> has associated with it
a sequence of part of speech tags, <img src="https://render.githubusercontent.com/render/math?math=POS_{1},%20POS_{2},...,%20POS_{T_{w}}"/>,
one tag for each occurrence of word <img src="https://render.githubusercontent.com/render/math?math=w"/>,
for a total of <img src="https://render.githubusercontent.com/render/math?math=T_{w}"/> tags for that word.
The task is to maximize the average log probability:

![\frac{1}{\left | W \right |}\sum_{w \in W}\sum_{t=1}^{T_{w}}log \ p(\textsc{pos}_{t}|w)](resources/eqn2.png)

## Commentary on learning POS embeddings with the word2vec approach

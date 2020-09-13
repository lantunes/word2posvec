# word2posvec

![Results](resources/brown.top5k.dim2.png)

Plot of learned word2posvec embeddings with 2 components. Red points are verbs, blue points are nouns, black points are 
numbers, green points are adjectives, and yellow points are adverbs.

![\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \leqslant j \leqslant c, j \neq 0}log \ p(w_{t+j}|w_{t})](resources/eqn1.png)

![\frac{1}{\left | W \right |}\sum_{w \in W}\sum_{s \in S_{w}}log \ p(s|w)](resources/eqn2.png)
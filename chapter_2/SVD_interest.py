import numpy as np
import corpus_init, co_occurence, improve_PMI
import matplotlib.pyplot as plt

text = 'You say goodbye and I say hello.'  # one special point symbol "." may change results
corpus, word_to_id, id_to_word = corpus_init.preprocess(text)
vocab_size = len(word_to_id)
C = co_occurence.create_co_matrix(corpus, vocab_size)

W = improve_PMI.PPMI(C)
np.set_printoptions(precision=3) # 3 siginificant digit

# S, V, D
U, S, V = np.linalg.svd(W)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

import numpy as np
import corpus_init, co_occurence

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2)+eps)  # normalize
    ny = y / np.sqrt(np.sum(y**2)+eps)  # normalize
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)  # simple check for security
        return
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)  # distance to every vector

    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue  # skip itself
        print('%s: %s' % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return
    


text = 'You say goodbye and I say hello.'  # one special point symbol "." may change results
corpus, word_to_id, id_to_word = corpus_init.preprocess(text)
vocab_size = len(word_to_id)
C = co_occurence.create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)





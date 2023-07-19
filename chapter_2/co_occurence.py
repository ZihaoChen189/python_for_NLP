import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = [word_to_id[w] for w in words]


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)  # the limit to the right content of every word 
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)  # the size of the co_matrix

    for ids, word_id in enumerate(corpus):
        # ids: every index in this corpus (related to the original sentence) first one? second one? third one?
        # word_id: the index value of these words in the original sentence one by one, IN the word_to_id or id_to_word
        # note: we could gain corpus through the defined preprocess(text) function
        for i in range(1, window_size+1):
            left_index = ids-i  # location index related to left content in the corpus
            right_index = ids+i  # location index related to right content in the corpus

            if left_index >= 0:  # had it reached the left edge of the sentence?
                left_word_id = corpus[left_index]  # neighborhood location index of the ids of "corpus"
                co_matrix[word_id, left_word_id] += 1  # draw in the co_matrix

            if right_index < corpus_size:  # corpus_size was used here (check right edge of the word in the sentence)
                right_word_id = corpus[right_index]  # neighborhood location index of the ids of "corpus"
                co_matrix[word_id, right_word_id] += 1  # draw in the co_matrix
    
    return co_matrix


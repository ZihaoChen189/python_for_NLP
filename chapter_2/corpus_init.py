import numpy as np

# text = "You say goodbye and I say hello."
# text = text.lower()
# text = text.replace('.', ' .')
# words = text.split(' ')

# word_to_id = {}
# id_to_word = {}
# for word in words:
#     if word not in word_to_id:
#         new_id = len(word_to_id)
#         id_to_word[new_id] = word
#         word_to_id[word] = new_id

# the place for every word"w" of the original sentence"words" on the word_to_id list to check their id s
# corpus = [word_to_id[w] for w in words]
# corpu = np.array(corpus)

# xs = [1, 2, 3, 4]
# imp_xs = [x**2 for x in xs]


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
    return corpus, word_to_id, id_to_word

# test:
text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)


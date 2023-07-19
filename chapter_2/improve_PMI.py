import numpy as np
import corpus_init, co_occurence

def PPMI(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)  # prepare for the output box
    N = np.sum(C)  # how many times did the word pair happen together?
    S = np.sum(C, axis=0)  # how many times does one single word happen on all the content
    total = C.shape[0] * C.shape[1]  # situation
    cnt = 0

    for i in range(C.shape[0]):
        for j in range (C.shape[1]):
            # interesting for (S[j]*S[i]), if brackets were cancelled, the reuslts would be diferent
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)  # information
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100+1) == 0:
                    print('%.lf%%done' % (100*cnt/total))

    return M

text = 'You say goodbye and I say hello.'  # one special point symbol "." may change results
corpus, word_to_id, id_to_word = corpus_init.preprocess(text)
vocab_size = len(word_to_id)
C = co_occurence.create_co_matrix(corpus, vocab_size)

W = PPMI(C)
np.set_printoptions(precision=3) # 3 siginificant digit

print('Covariance Matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

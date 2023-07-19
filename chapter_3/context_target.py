import numpy as np
import book_given_function
from tool import preprocess, softmax, SoftmaxWithLoss, MatMul


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in  # distribution representation

    def forward(self, contexts, target):
        # contexts example: (6, 2, 7) six matrix with (2, 7) shape
        h0 = self.in_layer0.forward(contexts[:, 0])  # first line of (2, 7) with all 6 examples
        h1 = self.in_layer1.forward(contexts[:, 1])  # seond line of (2, 7) with all 6 examples
        h = 0.5 * (h0+h1)  # average
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


text = 'You say goodbye and I say hello.' 
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
vocab_size = len(word_to_id)
target = book_given_function.convert_one_hot(target, vocab_size)
contexts = book_given_function.convert_one_hot(contexts, vocab_size)


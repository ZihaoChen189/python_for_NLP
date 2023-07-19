import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:  # simple change for one dimension data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


def softmax(x):
    help = np.max(x)
    exp_a = np.exp(x - help)  # slove nan question
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


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


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # save the loss
        self.y = None  # the softmax output
        self.t = None  # one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    

class MatMul:
    def __init__(self, W):
        self.params = [W]  # stroe the weight matrix which you wanted to learn
        self.grads = [np.zeros_like(W)]  # same shape with params
        self.x = None  # prepare for backward()

    def forward(self, x):
        W, = self.params
        output = np.dot(x, W)
        self.x = x  # prepare for backward()
        return output
    
    def backward(self, dout):
        W, = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)

        self.grads[0][...] = dW  # record the grads
        return dx
    

class MatMul:
    def __init__(self, W):
        self.params = [W]  # stroe the weight matrix which you wanted to learn
        self.grads = [np.zeros_like(W)]  # same shape with params
        self.x = None  # prepare for backward()

    def forward(self, x):
        W, = self.params
        output = np.dot(x, W)
        self.x = x  # prepare for backward()
        return output
    
    def backward(self, dout):
        W, = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)

        self.grads[0][...] = dW  # record the grads
        return dx
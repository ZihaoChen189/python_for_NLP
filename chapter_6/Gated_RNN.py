import numpy as np
import book_given_function
import pickle

def sigmoid(x):
    pass

class LSTM:
    def __init__(self, Wx, Wh, b):
        # Note Wx, Wh and b were all composed matrix with FOUR situations
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape  # they were friendly matrix
        # row: sample number; column: dimension, means how many pieces we need

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice of Four main parts
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        # np.hstack((a, b, c, d))
        pass
    

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        # prepare
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # whether store the intermediate states
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        # check the layer one by one
        for t in range(T):
            layer = LSTM(*self.params)  # unzip
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)  # h_prev and c_prev
            hs[:, t, :] = self.h  # update the output
            self.layers.append(layer)  # add this layer
        
        return hs
    
    def backward(self, dhs):
        # prepare
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):  # reversed()
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :]+dh, dc)  # execute backward operation
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad  # accumulate many words, not just one
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad  # Wx, Wh, b
            self.dh = dh
        
        return dxs
        
    def set_state(self, h, c=None):
        self.h, self.c = h, c
    
    def reset_state(self):
        self.h, self.c = None, None


class RnnLM:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D)/100).astype('f')
        lstm_Wx = (rn(D, 4*H)/np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            book_given_function.TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            book_given_function.TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = book_given_function.TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, file_name='RnnLM.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='RnnLM.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

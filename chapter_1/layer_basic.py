import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []
        # nothing to do with the W or b

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]  # [W, b]
        
    def forward(self, x):
        W, b = self.params
        output = np.dot(x, W) + b
        return output


# What about combining them ?
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params  # store weight/bias matrix one by one

    def predict(self, x):
        for layer in self.layers:
            # every layer would deal with this input "x", maybe do simple change or np.dot() with other parameters
            x = layer.forward(x)
        return x
    

# backward for repeat node:
# import numpy as np
# x = np.random.rand(1, 8)
# print(x)
# y = np.repeat(x, 7, axis=0)
# print(y)
# dy = np.random.randn(7, 8)
# dx = np.sum.repeat(dy, axis=0, keepdims=True)  # True: keep the shape of results as orginal one


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
    

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None  # prepare for the backward() layer

    def forward(self, x):
        output = 1/(1+np.exp(-x))
        self.out = output  # prepare for the backward() layer
        return output
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out)  * self.out  # self.out was used
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params
        output = np.dot(x, W) + b
        return output
    
    def backward(self, dout):
        W, b = self.params

        dx = np.dot(dout, W.T) + b
        dW = np.dot(self.x.T, dout) + b
        db = np.sum(dout, axis=0)  # also backward() of repeat node
        
        # update the grads of learning
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
    

class SGD:
    def __init__(self, lr):
        self.lr = lr

    def forward(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


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
    

def softmax(x):
    help = np.max(x)
    exp_a = np.exp(x - help)  # slove nan question
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:  # simple change for one dimension data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # just easy variable assignment
        I, H, O = input_size, hidden_size, output_size
        W1 = 0.01 * np.ranodm.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()  # this layer held different forward() and backward() operation

        self.params, self.grads = [], []
        # note: here, the last layer "SoftmaxWithLoss" didnt join 
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            return x  # prepare for the whole forward process
        
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)  # score->proba AND loss function
        return loss
    
    def backward(self, dout=1):  # note: start from dout=1
        dout = self.loss_layer.backward(dout)  # derive the value in the layers
        for layer in reversed(self.layers):
            dout = layer.backward(dout)  # the grads were updated one by one
        return dout

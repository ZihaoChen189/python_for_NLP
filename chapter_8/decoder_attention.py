import numpy as np

# this is a simple example of "hs" sum
# T, H = 5, 4
# hs = np.random.randn(T, H)
# a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])

# ar = a.reshape(5, 1).repeat(4, axis=1)
# print(ar.shape)

# t = hs * ar
# print(t.shape)

# c = np.sum(t, axis=0)
# print(c.shape)

# so, what about the mini-batch training:
# N, T, H = 10, 5, 4
# hs = np.random.randn(N, T, H)
# a = np.random.randn(N, T)
# ar = a.reshape(N, T, 1)

# t = hs * ar
# c = np.sum(t, axis=1)
# print(c.shape)


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)
        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)
        return dhs, da
    

class Softmax():
    def __init__(self):
        pass
    def forward(x):
        pass


# well, how to calculate the similarity?
# N, T, H = 10, 5, 4
# hs = np.random.randn(N, T, H)
# h = np.random.randn(N, H)
# hr = h.reshape(N, 1, H).repeat(T, axis=1)

# t = hs * hr
# print(t.shape)

# s = np.sum(t, axis=2)
# print(s.shape)

# softmax = Softmax()
# a = softmax.forward(s)

class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)
        
        self.cache = (hs, hr)
        return a
    
    def backward(self ,da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs

        dh = np.sum(dhr, axis=1)

        return dhs, dh

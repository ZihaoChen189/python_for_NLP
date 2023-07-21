# -*- coding: utf-8 -*-
import numpy as np
import book_given_function

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.params
        dW[...] = 0
        
        np.add(dW, self.idx, dout)
        return None
    

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None  # prepare for the backward()
        
    def forward(self, h, idx):  # h: intermediate neutrons
        target_W = self.embed.forward(idx)  # take that line out
        out = np.sum(target_W * h, axis=1)  # !!! NOT np.dot(), the one-dimension was what we want
        self.cache = (h, target_W)  # save the results, prepare for the backward()
        return out
    
    def backward(self, dout):
        h, target_W = self.cache  # get the cache and use them
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    

def cross_entropy_error(y, t):
    if y.ndim == 1:  # simple change for one dimension data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size

    
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = book_given_function.UnigramSampler(corpus, power, sample_size)
        
        # 1 positive sample + sample_size negatice samples
        self.loss_layers = [book_given_function.SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]
        
        self.grams, self.params = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # positive sample
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.nt32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # negative sample
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i+1].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)
            
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh
          
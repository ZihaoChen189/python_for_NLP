#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:36:19 2023

@author: robert
"""

import numpy as np
import book_given_function

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)
        
        self.cache = (x, h_prev, h_next)  # ready for the backward() operation
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        
        dt = dh_next * (1 - h_next**2)  # deivation of the tanh(x)

        db = np.sum(dt, axis=0)  # backward for the repeat node (gradient of the bias)
        # the first MatMul node
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        
        # the second MatMul node
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev
        
        
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        # stateful: whether we store the hidden state of each layer
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None  # so many layers that we needed
        # h:  the last hidden state of forward()
        # dh: ready for backward()
        self.h, self.dh = None, None
        self.stateful = stateful
        
    def set_state(self, h):
        self.h = h
        
    def reset_state(self):
        self.h = None
        
    # warning: this is not the forward betwen layers
    # is is the forward for STEPS of the input data
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
            
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
            
        return hs
        
    def backward(self, dhs):
        # get the arguments that we needed
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        
        # ready for the output
        dxs = np.empty((N, T, D), dtype='f')

        dh = 0  # useless for now, helpful to seq2seq
        grads = [0, 0, 0]  # Wx, Wh, b
        
        for t in reversed(range(T)):
            layer = self.layers[t]  # reversed layers
            
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # each sequence data by "t"
            dxs[:, t, :] = dx  # update the output
            
            for i, grad in enumerate(layer.grads):
                grads[i] += grad  # accumulate the argument of 3 places one by one in the Truncated BPTT operation
            
        for i, grad in enumerate(layer.grads):
            self.grads[i][...] = grad  # OK, update THREE gradient values
            
        self.dh = dh  # useless for now, helpful to seq2seq a non-zero value
        
        return dxs
    
    
class SimpleRnnLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        embed_W = (rn(V, D)/100).astype('f')
        rnn_Wx = (rn(D, H)/np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H)/np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.layers = [
            book_given_function.TimeEmbedding(embed_W),
            book_given_function.TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            book_given_function.TimeAffine(affine_W, affine_b)
            ]
        self.loss_layer = book_given_function.TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()
        
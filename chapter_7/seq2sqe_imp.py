import numpy as np
import book_given_function, decoder_imp, encoder_imp, Gated_RNN


class BaseModel:
    pass


class seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # get the value of the dimension
        V, D, H = vocab_size, wordvec_size, hidden_size
        # OK, use the implemented class for Encoder, Decoder and Softmax
        self.encoder = encoder_imp.Encoder(V, D, H)
        self.decoder = decoder_imp.Decoder(V, D, H)
        self.softmax = book_given_function.TimeSoftmaxWithLoss()
        # gradient update
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    # why we have ts: since it was the study operation
    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)  # the hidden state of the Encoder
        score = self.decoder.forward(decoder_xs, h)  # do the predict
        loss = self.softmax.forward(score, decoder_ts)  # check the loss, given the true label
        return loss
    
    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self, xs, start_id, sample_size):
        # well, this time is the real predict without the true label
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
    
# well, how to improve this model further?
# using special system: peeky decoder, which could make all layers in the decoder catch the hidden state of the encoder

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H+D, 4*H)/np.sqrt(H+D)).astype('f')
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        affine_W = (rn(H+H, V) / np.sqrt(H+H)).astpe('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = book_given_function.TimeEmbedding(embed_W)
        self.lstm = Gated_RNN.TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = book_given_function.TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)
        
        score = self.affine.forward(out)
        self.cache = H
        return score
    

class PeekySeq2Seq(seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = encoder_imp.Encoder(V, D, H)
        self.decoder = decoder_imp.Decoder(V, D, H)
        self.softmax = book_given_function.TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        

import numpy as np
import decoder_attention, attention_imp, encoder_imp, decoder_imp

class AttentionEncoder(encoder_imp.Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs  # this should be the WHOLE hidden states
    
    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
    
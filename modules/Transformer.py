import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
from modules.Encoder import Encoder
from modules.Decoder import Decoder
import modules.preprocess as preprocess


class Transformer(K.Model):
    def __init__(self, d_model: int, dff: int, num_heads: int, num_layers: int,
                 vocab_size: int, dropout: float = 0.1):
        super(Transformer, self).__init__(name="Transformer")

        self.enc_pad_mask = K.layers.Lambda(preprocess.create_pad_mask,
                                            output_shape=(1, 1, None), name="enc_pad_mask")

        self.look_ahead_mask = K.layers.Lambda(preprocess.create_look_ahead_mask,
                                               output_shape=(1, None, None), name="look_ahead_mask")

        self.dec_pad_mask = K.layers.Lambda(preprocess.create_pad_mask,
                                            output_shape=(1, 1, None), name="dec_pad_mask")

        self.encoder = Encoder(d_model, dff, num_heads, num_layers, vocab_size, dropout)
        self.decoder = Decoder(d_model, dff, num_heads, num_layers, vocab_size, dropout)

        self.ffnn = K.layers.Dense(vocab_size, name="outputs")

    def call(self, inputs, training=None, mask=None):
        input, dec_input = inputs["input"], inputs["dec_input"]

        enc_pad_mask = self.enc_pad_mask(input)

        look_ahead_mask = self.look_ahead_mask(dec_input)

        dec_pad_mask = self.dec_pad_mask(input)

        enc_out = self.encoder({"input": input, "mask": enc_pad_mask})

        dec_out = self.decoder({"input": dec_input, "encoder_output": enc_out,
                                "look_ahead_mask": look_ahead_mask, "padding_mask": dec_pad_mask})

        return self.ffnn(dec_out)

import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
from modules.Decoder.DecoderLayer import DecoderLayer
from ..preprocess.positional import PositionalEncoding


class Decoder(K.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int,
                 num_layers: int, vocab_size: int, dropout: int = 0.1):
        super(Decoder, self).__init__(name="decoder")

        self.d_model = d_model

        self.embd = K.layers.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(vocab_size, d_model)
        self.dropout = K.layers.Dropout(dropout)

        self.decoders = []

        for _ in range(num_layers):
            self.decoders.append(DecoderLayer(d_model, dff, num_heads, dropout))

    def call(self, inputs):
        input, encoder_output, look_ahead, padding_mask \
            = inputs["input"], inputs["encoder_output"], \
            inputs["look_ahead_mask"], inputs["padding_mask"]

        embeddings = self.embd(input)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_enc(embeddings)

        output = self.dropout(embeddings)

        for layers in self.decoders:
            output = layers({"input": output, "encoder_output": encoder_output,
                             "look_ahead_mask": look_ahead, "padding_mask": padding_mask})

        return output


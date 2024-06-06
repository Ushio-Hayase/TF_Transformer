import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
from modules.Encoder.EncoderLayer import EncoderLayer
from ..preprocess import positional


class Encoder(K.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int, num_layers: int, vocab_size: int, input_dim, dropout: int = 0.1):
        super(Encoder, self).__init__(name="Encoder")

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_ecd = positional.PositionalEncoding(vocab_size, d_model)
        self.dropout = K.layers.Dropout(dropout)

        self.encoder_layers = []

        for _ in range(num_layers):
            self.encoder_layers.append(EncoderLayer(d_model, dff, num_heads, dropout))

    def call(self, inputs):
        mask, input = inputs["mask"], inputs["input"]
        embeddings = self.pos_ecd(input)
        output = self.dropout(embeddings)

        for i in range(self.num_layers):
            output = self.encoder_layers[i]({"input": output, "mask": mask})

        return output

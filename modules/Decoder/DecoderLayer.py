import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
from ..QKV import MultiHeadAttention


class DecoderLayer(K.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int, dropout: int = 0.1):
        super(DecoderLayer, self).__init__(name="DecoderLayer")

        self.attn1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = K.layers.LayerNormalization()

        self.attn2 = MultiHeadAttention(d_model, num_heads)
        self.drop1 = K.layers.Dropout(dropout)
        self.norm2 = K.layers.LayerNormalization()

        self.ffnn1 = K.layers.Dense(dff, activation="relu")
        self.ffnn2 = K.layers.Dense(d_model)
        self.drop2 = K.layers.Dropout(dropout)
        self.norm3 = K.layers.LayerNormalization()

    def call(self, inputs):
        input, encoder_output, look_ahead, padding_mask =\
            inputs["input"], inputs["encoder_output"], \
            inputs["look_ahead_mask"], inputs["padding_mask"]

        attn1 = self.attn1({"query": input, "key": input, "value": input, "mask": look_ahead})
        attn1 = self.norm1(input + attn1)

        attn2 = self.attn2({"query": attn1, "key": encoder_output, "value": encoder_output,
                            "mask": padding_mask})
        attn2 = self.norm2(self.drop1(attn2) + attn1)

        output = self.ffnn1(attn2)
        output = self.ffnn2(output)

        output = self.drop2(output)
        output = self.norm3(output + attn2)

        return output



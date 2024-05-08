import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
from ..QKV import MultiHeadAttention


class EncoderLayer(K.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int, drop_out: int = 0.1):
        super(EncoderLayer, self).__init__(name="EncoderLayer")

        self.d_model = d_model

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = K.layers.Dropout(drop_out)
        self.norm1 = K.layers.LayerNormalization(epsilon=1e-6)

        self.ffnn1 = K.layers.Dense(dff, activation='relu')
        self.ffnn2 = K.layers.Dense(d_model)
        self.norm2 = K.layers.LayerNormalization()
        self.dropout2 = K.layers.Dropout(drop_out)

    def call(self, inputs):
        mask = inputs["mask"]
        input_data = inputs["input"]
        attention = self.attn({"query": input_data, "key": input_data, "value": input_data, "mask": mask})
        attention = self.dropout1(attention)
        attention = self.norm1(attention + input_data)

        output = self.ffnn1(attention)
        output = self.ffnn2(output)

        output = self.dropout2(output)
        output = self.norm2(attention + output)

        return output


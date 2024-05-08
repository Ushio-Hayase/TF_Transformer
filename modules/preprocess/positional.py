import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K


class PositionalEncoding(K.layers.Layer):
    def __init__(self, pos: int, d_model: int):
        super(PositionalEncoding, self).__init__(name="PositionalEncoding")
        self.pos_encoding = self.positional_encoding(pos, d_model)

    # 각 position 과 i에 따른 각도 구하기
    def get_theta(self, pos: tf.Tensor, i: tf.Tensor, d_model: int):
        return pos / tf.pow(10000, (i * 2) / d_model)

    def positional_encoding(self, pos, d_model):
        theta = self.get_theta(
            pos=B.arange(pos, dtype=tf.float32)[:, tf.newaxis],
            i=B.arange(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        sin = tf.sin(theta[:, 0::2]).numpy()  # 짝수에는 사인 적용
        cos = tf.cos(theta[:, 1::2]).numpy()  # 홀수에는 코사인 적용

        rad = tf.zeros(theta.shape).numpy()
        rad[:, 0::2] = sin
        rad[:, 1::2] = cos

        pos_encoding = B.constant(rad, dtype=tf.float32)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return pos_encoding

    def call(self, inputs: tf.Tensor):
        return inputs + self.pos_encoding[:, :B.shape(inputs)[1], :]



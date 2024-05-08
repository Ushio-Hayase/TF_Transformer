import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor | None):
    # Q, K의 전치 행렬의 내적 구하기
    matmul_qk = tf.linalg.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
    depth: tf.float32 = tf.cast(tf.shape(k)[-1], tf.float32)
    output = matmul_qk / tf.math.sqrt(depth)  # Q, K 내적의 K의 마지막 차원의 길이의 제곱근으로 나누기

    # 마스킹 있으면 마스크
    if mask is not None:
        output += (mask * -1e9)

    res = tf.nn.softmax(output, axis=-1)
    return tf.matmul(res, v)


class MultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__(name="MHA")
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0 # d_model을 num_heads로 나눈 나머지가 0이 아니면 오류

        self.depth = d_model // num_heads

        self.q_dense = K.layers.Dense(d_model)  # Q-가중치 ffnn
        self.k_dense = K.layers.Dense(d_model)  # K-가중치 ffnn
        self.v_dense = K.layers.Dense(d_model)  # V-가중치 ffnn

        self.out_dense = K.layers.Dense(d_model)  # 출력층 ffnn

    def split_heads(self, inputs: tf.Tensor, batch_size: int):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs["query"], inputs["key"], \
                        inputs["value"], inputs["mask"]

        batch_size = tf.shape(q)[0]

        # batch_size*seq_len*d_model 의 행렬로 변환
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention = scaled_dot_product_attention(q, k, v, mask=mask)  # batch_size*num_heads*sen_len*(d_model/num_heads)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # batch_size*sen_len*num_heads*(d_model/num_heads)

        output = tf.reshape(attention, [batch_size, -1, self.d_model])
        return self.out_dense(output)



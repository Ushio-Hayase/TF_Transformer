import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K

def create_pad_mask(x : tf.Tensor, pad_id: int = 1):
    mask = tf.cast(tf.math.equal(x, pad_id), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x : tf.Tensor):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), 0, -1)
    padding_mask = create_pad_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

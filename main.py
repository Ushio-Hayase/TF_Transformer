import tensorflow as tf

from modules.Transformer import Transformer
from data import Preprocess

epochs = 10
batch_size = 16
seq_len = 128

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_func(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 1))

    loss = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def metric(y_true, y_pred):
    pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)

    res = tf.equal(y_true, pred)
    acc = tf.reduce_mean(tf.cast(res, tf.float32))

    return acc

if __name__ == "__main__":
    gpus = tf.config.experimental.list_logical_devices('GPU')

    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('\n\n Running on multiple GPUs ', [gpu.name for gpu in gpus])

    with strategy.scope():
        model = Transformer(128, 512, 4,
                            4, 8002,  dropout = 0.1)
        dataset = Preprocess(batch_size, True)

        input = tf.keras.layers.Input(shape=128, batch_size=batch_size)
        dec_input = tf.keras.layers.Input(shape=128, batch_size=batch_size)

        model({"input": input, "dec_input": dec_input})

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        model.compile(loss=loss_func, optimizer='adam', metrics=[metric])

        model.fit(dataset, epochs=epochs, callbacks=[early_stopping])

        model.save("model")

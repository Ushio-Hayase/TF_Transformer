import os
import time
import tensorflow as tf

from modules.Transformer import Transformer
import numpy as np
from data import Dataloader

BATCH_SIZE = 64

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__ == "__main__":

    model = Transformer(256 , 512, 4,
                        4, 119547,BATCH_SIZE, dropout=0.1)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    learning_rate = CustomSchedule(256)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)


    x_train_df = np.load("x_train.npy") 
    y_train_df = np.load("y_train.npy")
    x_valid_df = np.load("x_valid.npy") 
    y_valid_df = np.load("y_valid.npy")


    train_dataloader = Dataloader(x_train_df, y_train_df, BATCH_SIZE)
    valid_dataloader = Dataloader(x_valid_df, y_valid_df, BATCH_SIZE)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

    checkpoint_path = "./checkpoints/train"

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function])

    inputs=[tf.keras.layers.Input(shape=(BATCH_SIZE,256)), tf.keras.layers.Input(shape=(BATCH_SIZE,256))]

    model.build(inputs)

    model.summary()

    model.fit(train_dataloader, batch_size=BATCH_SIZE, epochs=1, shuffle=True, validation_data=valid_dataloader)

    model.save_weights("model.weights.h5")
    model.save("model.keras")

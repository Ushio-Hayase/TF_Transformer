import time
import tensorflow as tf

from modules.Transformer import Transformer
import numpy as np
from data import Dataloader

CLS = 101 # bos token
SEP = 102 # eod token
PAD = 0 # pad_token
BATCH_SIZE = 64
EPOCH = 5


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


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

def train_step(inp, tar, train = True):
    pad_left = tf.constant([[CLS]], dtype=tf.int64)
    pad_left = tf.repeat(pad_left, BATCH_SIZE, axis=0)
    tar_inp = tar[:, 1:]
    tar_inp = tf.concat([pad_left, tar_inp], axis=-1)
    

    tar_real = tar
    mask = tf.equal(tar_real, PAD)
    indices = tf.where(mask)

    if tf.size(indices) > 0:
        first_index = indices[0]
        tar_real = tf.tensor_scatter_nd_update(tar_real, [first_index], [SEP])
    else:
        pad_right = tf.constant([[SEP]], dtype=tf.int64)
        pad_right = tf.repeat(pad_right, BATCH_SIZE, axis=0)
        tar_real = tar[:, :-1]
        tar_real = tf.concat([tar_real, pad_right], axis=-1)

    with tf.GradientTape() as tape:
        predictions = model([inp, tar_inp],
                                    training = True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if train:
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
    else:
        valid_loss(loss)
        valid_accuracy(accuracy_function(tar_real, predictions))

    return loss

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

@tf.function
def distributed_train_step(inp, tar, train = True):
    per_replica_losses = strategy.run(train_step, args=(inp, tar, train))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)


if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_logical_devices('GPU')

    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('\n\n Running on multiple GPUs ', [gpu.name for gpu in gpus])




    with strategy.scope():
        model = Transformer(256 , 512, 4,
                            4, 119547, dropout=0.1)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        valid_loss = tf.keras.metrics.Mean(name='val_loss')
        valid_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        learning_rate = CustomSchedule(256)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                            epsilon=1e-9)


        x_train_df = "" 
        y_train_df = ""
        x_valid_df = "" 
        y_valid_df = ""


        train_dataloader = Dataloader(x_train_df, y_train_df, BATCH_SIZE)
        vaild_dataloader = Dataloader(x_valid_df, y_valid_df, BATCH_SIZE)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(transformer=model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

        bar = tf.keras.utils.Progbar(train_dataloader.__len__(),stateful_metrics=['loss', 'acc'])

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')


        for epoch in range(EPOCH):
            start = time.time()

            train_loss.reset_state()
            train_accuracy.reset_state()
            valid_loss.reset_state()
            valid_accuracy.reset_state()

            for (batch, (inp, tar)) in enumerate(train_dataloader):
                distributed_train_step(inp, tar)

                bar.update(batch+1, values=[('loss', train_loss.result()), ("acc", train_accuracy.result())])

            for step, (x_batch_val, y_batch_val) in enumerate(vaild_dataloader):
                distributed_train_step(inp, tar)
                values = [('train_loss', train_loss.result()),\
                           ('train_acc', train_accuracy.result()), ('val_loss', valid_loss.result()),\
                              ('val_acc', valid_accuracy.result())]

                bar.update(step + 1, values=values, finalize=True)
            
            ckpt_save_path = ckpt_manager.save()
            model.save_weights(f'./checkpoints/{epoch}.weights.h5')
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

            print(f'Epoch : {epoch + 1} Loss : {train_loss.result():.4f} : Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    model.save_weghits("model.h5")
    model.save("model.keras")


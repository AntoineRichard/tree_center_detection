from numpy.core.fromnumeric import transpose
import tensorflow as tf
import numpy as np
import datetime
import argparse
import os 

from utils import videoSummary, videoSummaryCross, applyCorrection, augmentSequence
from samplers import SequenceSampler
from models import GRU_RESNET50

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--train_path',type=str)
    parser.add_argument('--val_path',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()

EPOCHS = 3000
BATCH_SIZE = 16

TRN_DS = SequenceSampler(args.train_path)
VAL_DS = SequenceSampler(args.val_path)

# LOAD Datasets

train_ds = tf.data.Dataset.range(2).interleave(
        lambda _: TRN_DS.getDataset(),
        num_parallel_calls=20
    )
train_ds = train_ds.map(lambda x, y: (applyCorrection(x), y), num_parallel_calls=20)
train_ds = train_ds.map(lambda x, y: augmentSequence(x, y), num_parallel_calls=20)
train_ds = train_ds.prefetch(1000)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(BATCH_SIZE)

val_ds = VAL_DS.getDataset()
val_ds = val_ds.map(lambda x, y: (applyCorrection(x), y), num_parallel_calls=20)
val_ds = val_ds.map(lambda x, y: augmentSequence(x, y), num_parallel_calls=20)
val_ds = val_ds.prefetch(1000)
val_ds = val_ds.batch(BATCH_SIZE)

# Load models

encoder, gru, direct = GRU_RESNET50(20,256,256)
encoder.summary()
gru.summary()
direct.summary()

# Creates training variables

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_gru_loss_metric = tf.keras.metrics.Mean('train_gru_loss', dtype=tf.float32)
train_direct_loss_metric = tf.keras.metrics.Mean('train_direct_loss', dtype=tf.float32)
val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
val_gru_loss_metric = tf.keras.metrics.Mean('val_gru_loss', dtype=tf.float32)
val_direct_loss_metric = tf.keras.metrics.Mean('val_direct_loss', dtype=tf.float32)
train_gru_error = tf.keras.metrics.Mean('train_gru_error', dtype=tf.float32)
train_direct_error = tf.keras.metrics.Mean('train_direct_error', dtype=tf.float32)
val_gru_error = tf.keras.metrics.Mean('val_gru_error', dtype=tf.float32)
val_direct_error = tf.keras.metrics.Mean('val_direct_error', dtype=tf.float32)

# Tf functions

@tf.function
def train_step(encoder, gru, direct, optimizer, x, y):
    with tf.GradientTape() as tape:
        embed = encoder(x, training=True)
        pgru = gru(embed, training=True)
        pdir = direct(embed, training=True)
        loss_gru = loss_object(y, pgru)
        loss_direct = loss_object(y, pdir)
        loss = loss_gru + loss_direct
    grads = tape.gradient(loss, encoder.trainable_variables + gru.trainable_variables + direct.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + gru.trainable_variables + direct.trainable_variables))
    train_gru_loss_metric.update_state(loss_gru)
    train_direct_loss_metric.update_state(loss_direct)
    train_loss_metric.update_state(loss)
    train_gru_error.update_state(tf.reduce_mean(tf.abs(pgru - y))*256)
    train_direct_error.update_state(tf.reduce_mean(tf.abs(pdir - y))*256)
    return pgru, pdir

@tf.function
def val_step(encoder, gru, direct, x, y):
    embed = encoder(x, training=True)
    pgru = gru(embed, training=True)
    pdir = direct(embed, training=True)
    loss_gru = loss_object(y, pgru)
    loss_direct = loss_object(y, pdir)
    loss = loss_gru + loss_direct
    val_gru_loss_metric.update_state(loss_gru)
    val_direct_loss_metric.update_state(loss_direct)
    val_loss_metric.update_state(loss)
    val_gru_error.update_state(tf.reduce_mean(tf.abs(pgru - y))*256)
    val_direct_error.update_state(tf.reduce_mean(tf.abs(pdir - y))*256)
    return pgru, pdir

# Tensorbard variables
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.join(args.output_path,'models',current_time), exist_ok=True)

train_log_dir = os.path.join(args.output_path,'tensorboard/' + current_time + '/train')
test_log_dir = os.path.join(args.output_path,'tensorboard/' + current_time + '/test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(test_log_dir)

step = 0
for epoch in range(EPOCHS):
    for x,y in train_ds:
        yg_, yd_ = train_step(encoder, gru, direct, optimizer, x, y)
        if (step % 50) == 0:
            template = 'Epoch {}, Step {}, total_loss {}, GRU loss {}, Direct loss {}'
            print(template.format(
                epoch,
                step,
                train_loss_metric.result(),
                train_gru_loss_metric.result(),
                train_direct_loss_metric.result()
            ))
        step += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss_metric.result(), step = step)
        tf.summary.scalar('gru_loss', train_gru_loss_metric.result(), step = step)
        tf.summary.scalar('direct_loss', train_direct_loss_metric.result(), step = step)
        videoSummary('input',np.array(x[:6]+0.5), step = step)
        videoSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
        videoSummaryCross('predictions_gru',np.array((x[:6]+0.5)*255).astype(np.uint8), (yg_+0.5)*256, step = step)
        videoSummaryCross('predictions_direct',np.array((x[:6]+0.5)*255).astype(np.uint8), (yd_+0.5)*256, step = step)
    train_loss_metric.reset_states()
    train_gru_loss_metric.reset_states()
    train_direct_loss_metric.reset_states()
    if (epoch % 10) == 0:
        for x,y in val_ds:
            yg_, yd_ = val_step(encoder, gru, direct, x, y)
        template = 'VALIDATION: Epoch {}, total_loss {}, GRU loss {}, Direct loss {}'
        print(template.format(
            epoch,
            step,
            val_loss_metric.result(),
            val_gru_loss_metric.result(),
            val_direct_loss_metric.result()
        ))
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step = step)
            tf.summary.scalar('gru_loss', val_gru_loss_metric.result(), step = step)
            tf.summary.scalar('direct_loss', val_direct_loss_metric.result(), step = step)
            videoSummary('input',np.array(x[:6]+0.5), step = step)
            videoSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
            videoSummaryCross('predictions_gru',np.array((x[:6]+0.5)*255).astype(np.uint8), (yg_+0.5)*256, step = step)
            videoSummaryCross('predictions_direct',np.array((x[:6]+0.5)*255).astype(np.uint8), (yd_+0.5)*256, step = step)
        val_loss_metric.reset_states()
        val_gru_loss_metric.reset_states()
        val_direct_loss_metric.reset_states()
        encoder.save(os.path.join(args.output_path,'models',current_time,'encoder_'+str(epoch)))
        gru.save(os.path.join(args.output_path,'models',current_time,'gru_'+str(epoch)))
        direct.save(os.path.join(args.output_path,'models',current_time,'direct_'+str(epoch)))

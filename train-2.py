from numpy.core.fromnumeric import transpose
import tensorflow as tf
import numpy as np
import datetime
import argparse
import os 

from utils import imageSummaryCross, applyCorrectionImage, augmentImage
from samplers import SimpleSampler
from models import VGG16_with_decoder, ResNet50_with_decoder, Simple_with_decoder

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str)
    parser.add_argument('--val_path',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()

# Args
EPOCHS = 100
BATCH_SIZE = 64

# LOAD Datasets
TRN_DS = SimpleSampler(args.train_path)
VAL_DS = SimpleSampler(args.val_path)
train_ds = tf.data.Dataset.range(2).interleave(
        lambda _: TRN_DS.getDataset(),
        num_parallel_calls=20
    )
train_ds = train_ds.map(lambda x, y: (applyCorrectionImage(x), y), num_parallel_calls=20)
train_ds = train_ds.cache('train_cache')
train_ds = train_ds.map(lambda x, y: augmentImage(x, y), num_parallel_calls=10)
train_ds = train_ds.prefetch(1000)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(BATCH_SIZE)

val_ds = VAL_DS.getDataset()
val_ds = val_ds.cache('val_cache')
val_ds = val_ds.map(lambda x, y: (applyCorrectionImage(x), y), num_parallel_calls=20)
val_ds = val_ds.map(lambda x, y: augmentImage(x, y), num_parallel_calls=10)
val_ds = val_ds.prefetch(1000)
val_ds = val_ds.batch(BATCH_SIZE)

# Load models
encoder, direct = Simple_with_decoder(256,256)
encoder.build((BATCH_SIZE, 256, 256, 1))
encoder.summary()
direct.summary()

# Creates training variables
loss_object = tf.keras.losses.MeanSquaredError()
direct_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
train_error = tf.keras.metrics.Mean('train_direct_error', dtype=tf.float32)
val_error = tf.keras.metrics.Mean('val_direct_error', dtype=tf.float32)

# Tf functions
@tf.function
def train_step(encoder, direct, direct_optimizer, x, y):
    with tf.GradientTape() as direct_tape:
        embed = encoder(x, training=True)
        pdir = direct(embed, training=True)
        loss = loss_object(y, pdir)
    direct_grads = direct_tape.gradient(loss, encoder.trainable_variables + direct.trainable_variables)
    direct_optimizer.apply_gradients(zip(direct_grads, encoder.trainable_variables + direct.trainable_variables))
        
    train_loss_metric.update_state(loss)
    train_error.update_state(tf.reduce_mean(tf.abs(pdir - y))*256)
    return pdir

@tf.function
def val_step(encoder, direct, x, y):
    embed = encoder(x, training=False)
    pdir = direct(embed, training=False)

    loss = loss_object(y, pdir)

    val_loss_metric.update_state(loss)
    val_error.update_state(tf.reduce_mean(tf.abs(pdir - y))*256)
    return pdir

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
        #print(x.shape,y.shape)
        yd_ = train_step(encoder, direct, direct_optimizer, x, y)
        if (step % 50) == 0:
            template = 'Epoch {}, Step {}, total_loss {}, error (pixels) {}'
            print(template.format(
                epoch,
                step,
                train_loss_metric.result(),
                train_error.result()
            ))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_metric.result(), step = step)
                tf.summary.scalar('error', train_error.result(), step = step)
            train_loss_metric.reset_states()
            train_error.reset_states()
        step += 1
    if (epoch % 1) == 0:
        with train_summary_writer.as_default():
            tf.summary.image('input',(np.array(x[:6]+0.5)*255).astype(np.uint8), step = step)
            imageSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
            imageSummaryCross('predictions_direct',np.array((x[:6]+0.5)*255).astype(np.uint8), (yd_+0.5)*256, step = step)
        for x,y in val_ds:
            yd_ = val_step(encoder, direct, x, y)
        template = 'VALIDATION: Epoch {}, Step {}, loss {}, error (pixels) {}'
        print(template.format(
            epoch,
            step,
            val_loss_metric.result(),
            val_error.result()
        ))
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step = step)
            tf.summary.scalar('error', val_error.result(), step = step)
            tf.summary.image('input',(np.array(x[:6]+0.5)*255).astype(np.uint8), step = step)
            imageSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
            imageSummaryCross('predictions_direct',np.array((x[:6]+0.5)*255).astype(np.uint8), (yd_+0.5)*256, step = step)
        val_loss_metric.reset_states()
        val_error.reset_states()
        encoder.save(os.path.join(args.output_path,'models',current_time,'encoder_'+str(epoch)))
        direct.save(os.path.join(args.output_path,'models',current_time,'direct_'+str(epoch)))

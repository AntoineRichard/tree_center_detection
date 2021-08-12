import tensorflow as tf
import numpy as np
import datetime
import argparse
import os 

from utils import videoSummary, videoSummaryCross, applyCorrection, augmentSequence
from model_utils import create_look_ahead_mask
from samplers import SequenceSampler
from models import Transformer

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str)
    parser.add_argument('--val_path',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()

EPOCHS = 3000
BATCH_SIZE = 16
SEQ_SIZE = 20

TRN_DS = SequenceSampler(args.train_path, seq_length=SEQ_SIZE)
VAL_DS = SequenceSampler(args.val_path, seq_length=SEQ_SIZE)

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

#encoder, gru, direct = GRU_RESNET50(20,256,256)
transformer = Transformer(num_layers=2, d_model=256, num_heads=4, dff=1024, target_shape=2, pe_input=2000, pe_target=2000)
# Load models
encoder = tf.keras.models.load_model('/home/gpu_user/antoine/WoodSeer/Xray_center_detection/models/20210802-080512/encoder_25')
out = encoder(np.ones((1,256,256,1)))
out = transformer(np.ones((16,20,256)), np.ones((16,20,2)), training=False, enc_padding_mask = None, look_ahead_mask=None, dec_padding_mask=None)
transformer.summary()
encoder.summary()

# Creates training variables
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
train_error = tf.keras.metrics.Mean('train_error', dtype=tf.float32)
val_error = tf.keras.metrics.Mean('val_error', dtype=tf.float32)

# Tf functions
@tf.function
def train_step(encoder, model, optimizer, x, y):
    B,T,H,W,C = x.shape
    x = tf.reshape(x,[B*T, H, W, C])
    embed = encoder(x, training=False)
    embed = tf.reshape(embed,[B, T, 256])
    look_ahead_mask = create_look_ahead_mask(tf.shape(y)[1])
    with tf.GradientTape() as model_tape:
        pred, _ = transformer(tf.stop_gradient(embed), y,
                                 True,
                                 None,
                                 look_ahead_mask,
                                 None)
        loss = loss_object(y, pred)
    grads = model_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    train_loss_metric.update_state(loss)
    train_error.update_state(tf.reduce_mean(tf.abs(pred - y))*256)
    return pred

@tf.function
def val_step(encoder, model, x, y):
    B,T,H,W,C = x.shape
    x = tf.reshape(x,[B*T, H, W, C])
    embed = encoder(x, training=False)
    embed = tf.reshape(embed,[B, T, 256])
    look_ahead_mask = create_look_ahead_mask(tf.shape(y)[1])
    pred, _ = transformer(tf.stop_gradient(embed), y,
                             False,
                             None,
                             look_ahead_mask,
                             None)
    loss = loss_object(y, pred)
    val_loss_metric.update_state(loss)
    val_error.update_state(tf.reduce_mean(tf.abs(pred - y))*256)
    return pred

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
        yg_ = train_step(encoder, transformer, optimizer, x, y)
        if (step % 50) == 0:
            template = 'Epoch {}, Step {}, loss {}, error {}'
            print(template.format(
                epoch,
                step,
                train_loss_metric.result(),
                train_error.result()
            ))
        step += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss_metric.result(), step = step)
        tf.summary.scalar('error', train_error.result(), step = step)
    train_loss_metric.reset_states()
    train_error.reset_states()
    if (epoch % 10) == 0:
        with train_summary_writer.as_default():
            videoSummary('input',np.array(x[:6]+0.5), step = step)
            videoSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
            videoSummaryCross('predictions_gru',np.array((x[:6]+0.5)*255).astype(np.uint8), (yg_+0.5)*256, step = step)
        for x,y in val_ds:
            yg_ = val_step(encoder, transformer, x, y)
        template = 'VALIDATION: Epoch {}, Step {}, loss {}, error {}'
        print(template.format(
            epoch,
            step,
            val_loss_metric.result(),
            val_error.result(),
        ))
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step = step)
            tf.summary.scalar('error', val_error.result(), step = step)
            videoSummary('input',np.array(x[:6]+0.5), step = step)
            videoSummaryCross('truth',np.array((x[:6]+0.5)*255).astype(np.uint8), (y+0.5)*256, step = step)
            videoSummaryCross('predictions_gru',np.array((x[:6]+0.5)*255).astype(np.uint8), (yg_+0.5)*256, step = step)
        val_loss_metric.reset_states()
        #transformer.save(os.path.join(args.output_path,'models',current_time,'transformer_'+str(epoch)))


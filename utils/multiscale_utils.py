from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

###############################################################################################################
# MULTISCALE CONVOLUTIONS
###############################################################################################################

class RegMultiScaleConvolution(tfkl.Layer):
    def __init__(self, d, kds=[1,3,6,9]):
        super().__init__()
        self.convs = [tfkl.Conv2D(d//4, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001), dilation_rate=(kd,kd)) for kd in kds]
        self.bns = [tfkl.BatchNormalization() for _ in kds]
        self.conv_merge = tfkl.Conv2D(d, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn_merge = tfkl.BatchNormalization()

    def call(self, x, training):
        cv = []
        for conv, bn in zip(self.convs,self.bns):
            tmp = conv(x)
            tmp = bn(tmp, training=training)
            cv.append(tf.nn.relu(tmp))
        out = tf.concat(cv,axis=-1)
        out = self.conv_merge(out)
        out = self.bn_merge(out)
        out = tf.nn.relu(out)
        return out

class MultiScaleConvolution(tfkl.Layer):
    def __init__(self, d, kds=[1,3,6,9]):
        super(MultiScaleConvolution, self).__init__()
        self.convs = [tfkl.Conv2D(d//4, (3,3), padding='same', dilation_rate=(kd,kd)) for kd in kds]
        self.conv_merge = tfkl.Conv2D(d, (3,3), padding='same')

    def call(self, x, training):
        cv = []
        for conv in self.convs:
            tmp = conv(x)
            cv.append(tf.nn.relu(tmp))
        out = tf.concat(cv,axis=-1)
        out = self.conv_merge(out)
        out = tf.nn.relu(out)
        return out

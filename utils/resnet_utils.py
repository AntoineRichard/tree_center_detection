from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

###############################################################################################################
# RESNET UTILS
###############################################################################################################

class ResidualBlock(Model):
    def __init__(self, f1, f2):
        super().__init__()
        self.conv1 = tfkl.Conv2D(f1, (1,1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(f1, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn2 = tfkl.BatchNormalization()
        self.conv3= tfkl.Conv2D(f2, (1,1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn3 = tfkl.BatchNormalization()

    def call(self, x, training):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        out += x
        out = tf.nn.leaky_relu(out)
        return out

class ResidualBlockDown(Model):
    def __init__(self, f1, f2, stride):
        super().__init__()
        self.conv1 = tfkl.Conv2D(f1, (1,1), (stride,stride), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(f1, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn2 = tfkl.BatchNormalization()
        self.conv3= tfkl.Conv2D(f2, (1,1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn3 = tfkl.BatchNormalization()
        self.conv_skip = tfkl.Conv2D(f2, (1,1), (stride, stride), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.bn_skip = tfkl.BatchNormalization()

    def call(self, x, training):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        skip = self.conv_skip(x)
        skip = self.bn_skip(skip)
        out += skip
        out = tf.nn.leaky_relu(out)
        return out

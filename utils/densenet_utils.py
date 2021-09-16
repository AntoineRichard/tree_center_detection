from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

###############################################################################################################
# DENSENET UTILS
###############################################################################################################

class DenseConnectedLayer(Model):
    def __init__(self, nb_filters, droprate=0.0, bottleneck=False):
        super().__init__()
        #self.bn1 = tfkl.BatchNormalization()
        self.bottleneck = bottleneck
        self.nb_filters = nb_filters
        if self.bottleneck:
            self.cv0 = tfkl.Conv2D(nb_filters*4, (1,1), padding='same', use_bias=False)
            if droprate > 0.0:
                self.allow_drop = True
                self.drop0 = tfkl.Dropout(droprate)
            else:
                self.allow_drop = False
            #self.bn0 = tfkl.BatchNormalization()
        self.cv1 = tfkl.Conv2D(nb_filters, (3,3), padding='same', use_bias=False)
        if droprate > 0.0:
            self.allow_drop = True
            self.drop1 = tfkl.Dropout(droprate)
        else:
            self.allow_drop = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.nb_filters)


    def call(self, x, training):
        #x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        if self.bottleneck:
            x = self.cv0(x)
            if self.allow_drop:
                x = self.drop0(x, training=training)
            #x = self.bn0(x, training)
            x = tf.nn.relu(x)
        x = self.cv1(x)
        if self.allow_drop:
            x = self.drop1(x, training=training)
        return x

class DenseBlock(Model):
    def __init__(self, nb_filters, growth_rate, num_layers, droprate=0.0, bottleneck=False, allow_growth=True):
        super().__init__()
        self.lays = []
        self.nb_filters = nb_filters
        for i in range(num_layers):
            self.lays.append(DenseConnectedLayer(growth_rate, droprate, bottleneck))
            if allow_growth:
                self.nb_filters += growth_rate

    def call(self, x, training):
        for layer in self.lays:
            tmp = layer(x, training)
            x = tf.concat([x, tmp],-1)
        return x

class DenseTransition(Model):
    def __init__(self, nb_filters, droprate=0.0, compression=1.0):
        super().__init__()
        self.conv = tfkl.Conv2D(int(nb_filters*compression),(1,1),padding='same', use_bias=False)
        #self.bn = tfkl.BatchNormalization()
        if droprate > 0.0:
            self.allow_drop = True
            self.drop = tfkl.Dropout()
        else:
            self.allow_drop = False
        self.pool = tfkl.AveragePooling2D((2,2), strides=(2,2))

    def call(self, x, training):
        #x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        if self.allow_drop:
            x = self.drop(x, training=training)
        x = self.pool(x)
        return x

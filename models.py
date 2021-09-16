from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfkl
import tensorflow as tf

from utils.transformer_utils import *
from utils.resnet_utils import *
from utils.densenet_utils import *
from utils.multiscale_utils import *

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                    target_shape, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_shape, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_shape)

    def call(self, inp, tar, training, enc_padding_mask, 
                look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder( tar, enc_output, training,
                                                     look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                    target_shape, pe_input, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                pe_input, rate)
        self.final_layer = tf.keras.layers.Dense(target_shape)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

class DenseNet(Model):
    def __init__(self, growthrate = 32, layers = [6, 12, 32, 32], num_init_feature = 64, reduction = 0.5, droprate=0.0, bottleneck = False, activation = tf.nn.relu, subsample_input = True):
        super(DenseNet, self).__init__()
        # Builds a dense net
        self.subsample_input = subsample_input
        self.act = activation
        # Compute compression rate
        self.compression = 1.0 - reduction
        # Growth rate and layers
        self.growthrate = growthrate
        self.blocks = layers

        # First layers, subdivide input if requested
        if self.subsample_input:
            self.cv0 = tfkl.Conv2D(num_init_feature, (7,7), strides=(2,2), padding='same', use_bias=False)
            #self.bn0 = tfkl.BatchNormalization()
            self.mp0 = tfkl.MaxPooling2D((3,3), strides = (2,2), padding='same')
        else:
            self.cv0 = tfkl.Conv2D(num_init_feature, (3,3), strides=(1,1), padding='same', use_bias=False)
        # Stacks dense blocks
        self.dense_blocks = []
        self.transitions = []
        self.depth = num_init_feature
        for layers in self.blocks[:-1]:
            blk = DenseBlock(self.depth, self.growthrate, layers, bottleneck = bottleneck, droprate = droprate)
            self.dense_blocks.append(blk)
            self.depth = blk.nb_filters
            self.transitions.append(DenseTransition(self.depth, droprate = droprate, compression = self.compression))
            self.depth = int(self.depth*self.compression)
        # No compression on last block
        blk = DenseBlock(self.depth, self.growthrate, layers, bottleneck = bottleneck, droprate = droprate)
        self.dense_blocks.append(blk)
        self.depth = blk.nb_filters
        self.transitions.append(DenseTransition(self.depth, droprate = droprate))
        # Tail
        #self.bn1 = tfkl.BatchNormalization()
        self.gap1 = tfkl.GlobalAveragePooling2D()
    
    def compute_output_shape(self, input_shape):
        return (self.depth)

    def call(self, x, training):
        # Call intial layers
        x = self.cv0(x)
        if self.subsample_input:
            #x = self.bn0(x, training)
            x = self.act(x)
            x = self.mp0(x)
        # Apply blocks
        for blk, tra in zip(self.dense_blocks, self.transitions):
            x = blk(x, training)
            x = tra(x, training)
        # End of encoder
        #x = self.bn1(x, training)
        x = self.act(x)
        x = self.gap1(x)
        return x


class ResNet50(Model):
    def __init__(self, depth):
        super(ResNet50, self).__init__()
        self.depth = depth
        # Init
        self.a1 = tfkl.Conv2D(depth//32, (7,7), (2,2), padding='same')
        self.bn1 = tfkl.BatchNormalization()
        self.p1 = tfkl.MaxPooling2D()
        # Block 1
        self.b1 = ResidualBlockDown(depth//32, depth//8, 2)
        self.b2 = ResidualBlock(depth//32, depth//8)
        self.b3 = ResidualBlock(depth//32, depth//8)
        # Block 2
        self.c1 = ResidualBlockDown(depth//16, depth//4 ,2)
        self.c2 = ResidualBlock(depth//16, depth//4)
        self.c3 = ResidualBlock(depth//16, depth//4)
        self.c4 = ResidualBlock(depth//16, depth//4)
        # Block 3
        self.d1 = ResidualBlockDown(depth//8, depth//2, 2)
        self.d2 = ResidualBlock(depth//8, depth//2)
        self.d3 = ResidualBlock(depth//8, depth//2)
        self.d4 = ResidualBlock(depth//8, depth//2)
        self.d5 = ResidualBlock(depth//8, depth//2)
        # Block 4
        self.e1 = ResidualBlockDown(depth//4, depth, 2)
        self.e2 = ResidualBlock(depth//4, depth)
        self.e3 = ResidualBlock(depth//4, depth)
        # Dense
        self.p2 = tfkl.AveragePooling2D((2, 2), padding='same')
        self.flat = tfkl.Flatten()
        self.fc1 = tfkl.Dense(depth//4, activation=tf.nn.leaky_relu)

    def compute_output_shape(self, input_shape):
        return (self.depth//4)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        # Init
        out = self.a1(x)
        out = self.bn1(out)
        out = tf.nn.leaky_relu(out)
        out = self.p1(out)
        # Block 1
        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        # Block 2
        out = self.c1(out)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        # Block 3
        out = self.d1(out)
        out = self.d2(out)
        out = self.d3(out)
        out = self.d4(out)
        out = self.d5(out)
        # Block 4
        out = self.e1(out)
        out = self.e2(out)
        out = self.e3(out)
        # Dense
        out = self.p2(out)
        out = self.flat(out)
        out = self.fc1(out)
        return out

class VGG16Drop(Model):
    def __init__(self, depth, rate = 0.15):
        super().__init__()
        self.depth = depth
        self.rate = rate
        #self.drop = tfkl.Dropout(self.rate)
        # Block 1
        self.c11 = tfkl.Conv2D(depth//16, (3,3), padding='same', name='c11')
        self.bn11 = tfkl.BatchNormalization(name='bn11')
        self.c12 = tfkl.Conv2D(depth//16, (3,3), padding='same', name='c12')
        self.bn12 = tfkl.BatchNormalization(name='bn12')
        self.p1 = tfkl.MaxPooling2D((2, 2), padding='same', name='p1')
        # Block 2
        self.c21 = tfkl.Conv2D(depth//8, (3,3), padding='same', name='c21')
        self.bn21 = tfkl.BatchNormalization(name='bn21')
        self.c22 = tfkl.Conv2D(depth//8, (3,3), padding='same',name='c22')
        self.bn22 = tfkl.BatchNormalization(name='bn22')
        self.p2 = tfkl.MaxPooling2D((2, 2), padding='same', name='p2')
        # Block 3
        self.c31 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c31')
        self.bn31 = tfkl.BatchNormalization(name='bn31')
        self.c32 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c32')
        self.bn32 = tfkl.BatchNormalization(name='bn32')
        self.c33 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c33')
        self.bn33 = tfkl.BatchNormalization(name='bn33')
        self.p3 = tfkl.MaxPooling2D((2, 2), padding='same',name='p3')
        # Block 4
        self.c41 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c41')
        self.bn41 = tfkl.BatchNormalization(name='bn41')
        self.c42 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c42')
        self.bn42 = tfkl.BatchNormalization(name='bn42')
        self.c43 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c43')
        self.bn43 = tfkl.BatchNormalization(name='bn43')
        self.p4 = tfkl.MaxPooling2D((2, 2), padding='same',name='p4')
        # Block 5
        self.c51 = tfkl.Conv2D(depth, (3,3), padding='same',name='c51')
        self.bn51 = tfkl.BatchNormalization(name='bn51')
        self.c52 = tfkl.Conv2D(depth, (3,3), padding='same',name='c52')
        self.bn52 = tfkl.BatchNormalization(name='bn52')
        self.c53 = tfkl.Conv2D(depth, (3,3), padding='same',name='c53')
        self.bn53 = tfkl.BatchNormalization(name='bn53')
        self.p5 = tfkl.MaxPooling2D((2, 2), padding='same',name='p5')
        # Dense
        self.flat = tfkl.Flatten()
        self.fc1 = tfkl.Dense(depth//4, activation='relu',name='d1')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.depth//4)
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x, training):
        # Block 1
        out = self.c11(x)
        out = self.bn11(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c12(out)
        out = self.bn12(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.p1(out)
        # Block 2
        out = self.c21(out)
        out = self.bn21(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c22(out)
        out = self.bn22(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.p2(out)
        # Block 3
        out = self.c31(out)
        out = self.bn31(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c32(out)
        out = self.bn32(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c33(out)
        out = self.bn33(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.p3(out)
        # Block 4
        out = self.c41(out)
        out = self.bn41(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c42(out)
        out = self.bn42(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c43(out)
        out = self.bn43(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.p4(out)
        # Block 5
        out = self.c51(out)
        out = self.bn51(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c52(out)
        out = self.bn52(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.c53(out)
        out = self.bn53(out, training)
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, rate=self.rate)
        #out = self.drop(out, training)
        out = self.p5(out)
        # Dense
        out = self.flat(out)
        out = self.fc1(out)
        return out

class VGG16(Model):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        # Block 1
        self.c11 = tfkl.Conv2D(depth//16, (3,3), padding='same', name='c11')
        self.bn11 = tfkl.BatchNormalization(name='bn11')
        self.c12 = tfkl.Conv2D(depth//16, (3,3), padding='same', name='c12')
        self.bn12 = tfkl.BatchNormalization(name='bn12')
        self.p1 = tfkl.MaxPooling2D((2, 2), padding='same', name='p1')
        # Block 2
        self.c21 = tfkl.Conv2D(depth//8, (3,3), padding='same', name='c21')
        self.bn21 = tfkl.BatchNormalization(name='bn21')
        self.c22 = tfkl.Conv2D(depth//8, (3,3), padding='same',name='c22')
        self.bn22 = tfkl.BatchNormalization(name='bn22')
        self.p2 = tfkl.MaxPooling2D((2, 2), padding='same', name='p2')
        # Block 3
        self.c31 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c31')
        self.bn31 = tfkl.BatchNormalization(name='bn31')
        self.c32 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c32')
        self.bn32 = tfkl.BatchNormalization(name='bn32')
        self.c33 = tfkl.Conv2D(depth//4, (3,3), padding='same',name='c33')
        self.bn33 = tfkl.BatchNormalization(name='bn33')
        self.p3 = tfkl.MaxPooling2D((2, 2), padding='same',name='p3')
        # Block 4
        self.c41 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c41')
        self.bn41 = tfkl.BatchNormalization(name='bn41')
        self.c42 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c42')
        self.bn42 = tfkl.BatchNormalization(name='bn42')
        self.c43 = tfkl.Conv2D(depth//2, (3,3), padding='same',name='c43')
        self.bn43 = tfkl.BatchNormalization(name='bn43')
        self.p4 = tfkl.MaxPooling2D((2, 2), padding='same',name='p4')
        # Block 5
        self.c51 = tfkl.Conv2D(depth, (3,3), padding='same',name='c51')
        self.bn51 = tfkl.BatchNormalization(name='bn51')
        self.c52 = tfkl.Conv2D(depth, (3,3), padding='same',name='c52')
        self.bn52 = tfkl.BatchNormalization(name='bn52')
        self.c53 = tfkl.Conv2D(depth, (3,3), padding='same',name='c53')
        self.bn53 = tfkl.BatchNormalization(name='bn53')
        self.p5 = tfkl.MaxPooling2D((2, 2), padding='same',name='p5')
        # Dense
        self.flat = tfkl.Flatten()
        self.fc1 = tfkl.Dense(depth//4, activation='relu',name='d1')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.depth//4)
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x):
        # Block 1
        out = self.c11(x)
        out = self.bn11(out)
        out = tf.nn.relu(out)
        out = self.c12(out)
        out = self.bn12(out)
        out = tf.nn.relu(out)
        out = self.p1(out)
        # Block 2
        out = self.c21(out)
        out = self.bn21(out)
        out = tf.nn.relu(out)
        out = self.c22(out)
        out = self.bn22(out)
        out = tf.nn.relu(out)
        out = self.p2(out)
        # Block 3
        out = self.c31(out)
        out = self.bn31(out)
        out = tf.nn.relu(out)
        out = self.c32(out)
        out = self.bn32(out)
        out = tf.nn.relu(out)
        out = self.c33(out)
        out = self.bn33(out)
        out = tf.nn.relu(out)
        out = self.p3(out)
        # Block 4
        out = self.c41(out)
        out = self.bn41(out)
        out = tf.nn.relu(out)
        out = self.c42(out)
        out = self.bn42(out)
        out = tf.nn.relu(out)
        out = self.c43(out)
        out = self.bn43(out)
        out = tf.nn.relu(out)
        out = self.p4(out)
        # Block 5
        out = self.c51(out)
        out = self.bn51(out)
        out = tf.nn.relu(out)
        out = self.c52(out)
        out = self.bn52(out)
        out = tf.nn.relu(out)
        out = self.c53(out)
        out = self.bn53(out)
        out = tf.nn.relu(out)
        out = self.p5(out)
        # Dense
        out = self.flat(out)
        out = self.fc1(out)
        return out

class MS_VGG16(Model):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        # Block 1
        self.c11 = MultiScaleConvolution(depth//16, [1,3,6,9,12])
        self.c12 = MultiScaleConvolution(depth//16, [1,3,6,9,12])
        self.p1 = tfkl.MaxPooling2D((2, 2), padding='same', name='p1')
        # Block 2
        self.c21 = MultiScaleConvolution(depth//8, [1,3,6,9,12])
        self.c22 = MultiScaleConvolution(depth//8, [1,3,6,9,12])
        self.p2 = tfkl.MaxPooling2D((2, 2), padding='same', name='p2')
        # Block 3
        self.c31 = MultiScaleConvolution(depth//4, [1,3,6,9])
        self.c32 = MultiScaleConvolution(depth//4, [1,3,6,9])
        self.p3 = tfkl.MaxPooling2D((2, 2), padding='same',name='p3')
        # Block 4
        self.c41 = MultiScaleConvolution(depth//2, [1,3,6,9])
        self.c42 = MultiScaleConvolution(depth//2, [1,3,6,9])
        self.p4 = tfkl.MaxPooling2D((2, 2), padding='same',name='p4')
        # Block 5
        self.c51 = MultiScaleConvolution(depth, [1,3,6])
        self.c52 = MultiScaleConvolution(depth, [1,3,6])
        self.p5 = tfkl.MaxPooling2D((2, 2), padding='same',name='p5')
        # Dense
        self.flat = tfkl.Flatten()
        self.fc1 = tfkl.Dense(depth//2, activation='relu',name='d1')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.depth//2)
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x, training):
        # Block 1
        out = self.c11(x, training)
        out = self.c12(out, training)
        out = self.p1(out)
        # Block 2
        out = self.c21(out, training)
        out = self.c22(out, training)
        out = self.p2(out)
        # Block 3
        out = self.c31(out, training)
        out = self.c32(out, training)
        out = self.p3(out)
        # Block 4
        out = self.c41(out, training)
        out = self.c42(out, training)
        out = self.p4(out)
        # Block 5
        out = self.c51(out, training)
        out = self.c52(out, training)
        out = self.p5(out)
        # Dense
        out = self.flat(out)
        out = self.fc1(out)
        return out

def BidirGRU(seq_size=20, input_size=256, output=2, neurons=256):
    gru = Sequential()
    gru.add(tfkl.TimeDistributed(tfkl.Dense(neurons, activation='relu'), input_shape=(seq_size, input_size)))
    forward_layer_1 = tfkl.GRU(neurons, return_sequences=True)
    backward_layer_1 = tfkl.GRU(neurons, return_sequences=True, go_backwards=True)
    gru.add(tfkl.Bidirectional(forward_layer_1, backward_layer=backward_layer_1))
    gru.add(tfkl.TimeDistributed(tfkl.Dense(neurons, activation='relu')))
    gru.add(tfkl.TimeDistributed(tfkl.Dropout(rate=0.1)))
    gru.add(tfkl.TimeDistributed(tfkl.Dense(neurons//2, activation='relu')))
    gru.add(tfkl.TimeDistributed(tfkl.Dense(output)))
    return gru

def BidirLSTM(seq_size=20, input_size=256, output=2, neurons=256):
    lstm = Sequential()
    lstm.add(tfkl.TimeDistributed(tfkl.Dense(neurons, activation='relu'), input_shape=(seq_size, input_size)))
    forward_layer_1 = tfkl.LSTM(neurons, return_sequences=True)
    backward_layer_1 = tfkl.LSTM(neurons, return_sequences=True, go_backwards=True)
    lstm.add(tfkl.Bidirectional(forward_layer_1, backward_layer=backward_layer_1))
    lstm.add(tfkl.TimeDistributed(tfkl.Dense(neurons, activation='relu')))
    lstm.add(tfkl.TimeDistributed(tfkl.Dropout(rate=0.1)))
    lstm.add(tfkl.TimeDistributed(tfkl.Dense(neurons//2, activation='relu')))
    lstm.add(tfkl.TimeDistributed(tfkl.Dense(output)))
    return lstm

def ResNet50_with_decoder(height, width, output=2, depth=2048):
    encoder = ResNet50(depth)
    model2 = Sequential()
    model2.add(tfkl.Dense(256, activation='relu', input_shape=[depth//4]))
    model2.add(tfkl.Dropout(rate=0.1))
    model2.add(tfkl.Dense(256, activation='relu'))
    model2.add(tfkl.Dense(output))
    return encoder, model2

def VGG16_with_decoder(height, width, output=2, depth=1024):
    encoder = VGG16Drop(depth)
    model2 = Sequential(name="dense_decoder")
    model2.add(tfkl.Dense(128, activation='relu', input_shape=[depth//4]))
    model2.add(tfkl.Dropout(rate=0.1))
    model2.add(tfkl.Dense(128, activation='relu'))
    model2.add(tfkl.Dense(output))
    return encoder, model2

def Simple_with_decoder(height, width, output=2, depth=1024):
    encoder = Sequential(name="encoder")
    encoder.add(tfkl.Conv2D(32, (3,3), padding = 'same', activation='relu', input_shape=[height,width,1]))
    encoder.add(tfkl.Conv2D(32, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tfkl.Conv2D(64, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.Conv2D(64, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tfkl.Conv2D(128, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.Conv2D(128, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tfkl.Conv2D(256, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.Conv2D(256, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tfkl.Conv2D(256, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.Conv2D(256, (3,3), padding = 'same', activation='relu'))
    encoder.add(tfkl.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tfkl.Flatten())
    encoder.add(tfkl.Dense(256, activation='relu'))

    model2 = Sequential(name="dense_decoder")
    model2.add(tfkl.Dense(128, activation='relu', input_shape=[256]))
    model2.add(tfkl.Dropout(rate=0.1))
    model2.add(tfkl.Dense(128, activation='relu'))
    model2.add(tfkl.Dense(output))
    return encoder, model2

def DenseNet_with_decoder(height, width, output=2, depth=256):
    encoder = DenseNet()

    model2 = Sequential(name="dense_decoder")
    model2.add(tfkl.Dense(128, activation='relu', input_shape=[encoder.depth]))
    model2.add(tfkl.Dropout(rate=0.1))
    model2.add(tfkl.Dense(128, activation='relu'))
    model2.add(tfkl.Dense(output))
    return encoder, model2

def VGG16MS_with_decoder(height, width, output=2, depth=512):
    encoder = MS_VGG16(depth)
    model2 = Sequential(name="dense_decoder")
    model2.add(tfkl.Dense(256, activation='relu', input_shape=[depth//2]))
    model2.add(tfkl.Dropout(rate=0.1))
    model2.add(tfkl.Dense(256, activation='relu'))
    model2.add(tfkl.Dense(output))
    return encoder, model2


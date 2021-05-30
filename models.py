from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfkl
import tensorflow as tf

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
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        out += x
        out = tf.nn.relu(out)
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
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        skip = self.conv_skip(x)
        skip = self.bn_skip(skip)
        out += skip
        out = tf.nn.relu(out)
        return out

class ResNet50(Model):
    def __init__(self, depth):
        super().__init__()
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
        self.fc1 = tfkl.Dense(depth//4, activation='relu')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.depth//4)

    
    def build(self, input_shape):
        super().build(input_shape)

        
    def call(self, x):
        # Init
        out = self.a1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)
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

class VGG16(Model):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        # Block 1
        self.c11 = tfkl.Conv2D(depth//16, (3,3), padding='same')
        self.bn11 = tfkl.BatchNormalization()
        self.c12 = tfkl.Conv2D(depth//16, (3,3), padding='same')
        self.bn12 = tfkl.BatchNormalization()
        self.p1 = tfkl.MaxPooling2D((2, 2), padding='same')
        # Block 2
        self.c21 = tfkl.Conv2D(depth//8, (3,3), padding='same')
        self.bn21 = tfkl.BatchNormalization()
        self.c22 = tfkl.Conv2D(depth//8, (3,3), padding='same')
        self.bn22 = tfkl.BatchNormalization()
        self.p2 = tfkl.MaxPooling2D((2, 2), padding='same')
        # Block 3
        self.c31 = tfkl.Conv2D(depth//4, (3,3), padding='same')
        self.bn31 = tfkl.BatchNormalization()
        self.c32 = tfkl.Conv2D(depth//4, (3,3), padding='same')
        self.bn32 = tfkl.BatchNormalization()
        self.c33 = tfkl.Conv2D(depth//4, (3,3), padding='same')
        self.bn33 = tfkl.BatchNormalization()
        self.p3 = tfkl.MaxPooling2D((2, 2), padding='same')
        # Block 4
        self.c41 = tfkl.Conv2D(depth//2, (3,3), padding='same')
        self.bn41 = tfkl.BatchNormalization()
        self.c42 = tfkl.Conv2D(depth//2, (3,3), padding='same')
        self.bn42 = tfkl.BatchNormalization()
        self.c43 = tfkl.Conv2D(depth//2, (3,3), padding='same')
        self.bn43 = tfkl.BatchNormalization()
        self.p4 = tfkl.MaxPooling2D((2, 2), padding='same')
        # Block 5
        self.c51 = tfkl.Conv2D(depth, (3,3), padding='same')
        self.bn51 = tfkl.BatchNormalization()
        self.c52 = tfkl.Conv2D(depth, (3,3), padding='same')
        self.bn52 = tfkl.BatchNormalization()
        self.c53 = tfkl.Conv2D(depth, (3,3), padding='same')
        self.bn53 = tfkl.BatchNormalization()
        self.p5 = tfkl.MaxPooling2D((2, 2), padding='same')
        # Dense
        self.flat = tfkl.Flatten()
        self.fc1 = tfkl.Dense(depth//4, activation='relu')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.depth//4)
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x):
        # Block 1
        out = self.c11(x)
        out = self.bn11(out)
        out = self.c12(x)
        out = self.bn12(out)
        out = self.p1(out)
        # Block 2
        out = self.c21(out)
        out = self.bn21(out)
        out = self.c22(out)
        out = self.bn22(out)
        out = self.p2(out)
        # Block 3
        out = self.c31(out)
        out = self.bn31(out)
        out = self.c32(out)
        out = self.bn32(out)
        out = self.c33(out)
        out = self.bn33(out)
        out = self.p3(out)
        # Block 4
        out = self.c41(out)
        out = self.bn41(out)
        out = self.c42(out)
        out = self.bn42(out)
        out = self.c43(out)
        out = self.bn43(out)
        out = self.p4(out)
        # Block 5
        out = self.c51(out)
        out = self.bn51(out)
        out = self.c52(out)
        out = self.bn52(out)
        out = self.c53(out)
        out = self.bn53(out)
        out = self.p5(out)
        # Dense
        out = self.flat(out)
        out = self.fc1(out)
        return out


def GRU_RESNET50(seq_size, height, width, output=2, depth=1024):
    encoder = Sequential()
    encoder.add(tfkl.TimeDistributed(ResNet50(depth), input_shape=(seq_size, height, width, 1)))
    model1 = Sequential()
    forward_layer_1 = tfkl.GRU(depth//4, return_sequences=True)
    backward_layer_1 = tfkl.GRU(depth//4, return_sequences=True, go_backwards=True)
    model1.add(tfkl.Bidirectional(forward_layer_1, backward_layer=backward_layer_1, input_shape=(seq_size, depth//4)))
    model1.add(tfkl.TimeDistributed(tfkl.Dense(128, activation='relu')))
    model1.add(tfkl.TimeDistributed(tfkl.Dense(output)))
    model2 = Sequential()
    model2.add(tfkl.TimeDistributed(tfkl.Dense(128, activation='relu'), input_shape=(seq_size, depth//4)))
    model2.add(tfkl.TimeDistributed(tfkl.Dense(128, activation='relu')))
    model2.add(tfkl.TimeDistributed(tfkl.Dense(output)))
    return encoder, model1, model2

import tensorflow as tf
import tensorflow.keras.layers as keras_layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, AveragePooling2D, UpSampling2D, Conv2D, BatchNormalization, concatenate, Reshape, Dense, multiply, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.activations import relu, sigmoid
from typing import List, Tuple
from cnn_architectures.utils.common import ConvBlock


class ASPP(Layer):
    def __init__(self, filters, input_shape, name: str = None):
        super(ASPP, self).__init__(name=name)
        self.__filters = filters
        self.__input_shape = input_shape

        self.avgpool = AveragePooling2D(pool_size=(input_shape[1], input_shape[2]))
        self.up = UpSampling2D(size=(input_shape[1], input_shape[2]), interpolation='bilinear')
        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            dilation_rate=1,
                            padding='same',
                            use_bias=False)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=6,
                            padding='same',
                            use_bias=False)
        self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=12,
                            padding='same',
                            use_bias=False)
        self.bn4 = BatchNormalization()

        self.conv5 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=18,
                            padding='same',
                            use_bias=False)
        self.bn5 = BatchNormalization()

        self.conv = Conv2D(filters=filters,
                           kernel_size=(1, 1),
                           padding='same',
                           use_bias=False)
        self.bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        y1 = self.avgpool(inputs)
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = relu(y1)
        y1 = self.up(y1)

        y2 = self.conv2(inputs)
        y2 = self.bn2(y2)
        y2 = relu(y2)

        y3 = self.conv3(inputs)
        y3 = self.bn3(y3)
        y3 = relu(y3)

        y4 = self.conv4(inputs)
        y4 = self.bn4(y4)
        y4 = relu(y4)

        y5 = self.conv5(inputs)
        y5 = self.bn5(y5)
        y5 = relu(y5)
        y = concatenate([y1, y2, y3, y4, y5])

        y = self.conv(y)
        y = self.bn(y)
        y = relu(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'input_shape': self.__input_shape,
        })
        return config


class ForwardConnectedDecoder(Layer):

    def __init__(self, input_size, connections: List[keras_layer.Layer], filter_sizes: Tuple[int] = (256, 128, 64, 32), name: str = None, **kwargs):
        super(ForwardConnectedDecoder, self).__init__(name=name, **kwargs)
        self.__input_size = input_size

        self.filter_sizes = filter_sizes

        self.connections = connections
        self.upsamplings: List[UpSampling2D] = []
        self.concatenations: List[concatenate] = []
        self.convolutions: List[ConvolutionalBlock] = []

        for index, filter_size in enumerate(self.filter_sizes):
            upsampling2d = UpSampling2D((2, 2), interpolation='bilinear')
            self.upsamplings.append(upsampling2d)

            concat = concatenate
            self.concatenations.append(concat)

            conv = ConvolutionalBlock(filters=filter_size)
            self.convolutions.append(conv)

    def call(self, inputs, **kwargs):

        skip_connections = self.connections.copy()
        skip_connections.reverse()
        x = inputs

        for index, filter_size in enumerate(self.filter_sizes):
            x = self.upsamplings[index](x)
            x = self.concatenations[index]([x, skip_connections[index]])
            x = self.convolutions[index](x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections': self.connections,
            'filter_sizes': self.filter_sizes,
            'input_size': self.__input_size
        })
        return config


class ForwardEncoder(Layer):

    def __init__(self, filter_sizes: Tuple[int] = (32, 64, 128, 256), name: str = None, **kwargs):
        super(ForwardEncoder, self).__init__(name=name, **kwargs)

        self.filter_sizes = filter_sizes

        self.convolutions: List[ConvolutionalBlock] = []
        self.pools: List[MaxPool2D] = []
        self.skipped_connections = []

        for index, filter_size in enumerate(self.filter_sizes):
            conv = ConvolutionalBlock(filters=filter_size)
            self.convolutions.append(conv)

            max_pool = MaxPool2D((2, 2))
            self.pools.append(max_pool)

    def call(self, inputs, **kwargs):
        x = inputs

        for index, filter_size in enumerate(self.filter_sizes):
            x = self.convolutions[index](x)
            self.skipped_connections.append(x)
            x = self.pools[index](x)

        return x, self.skipped_connections

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_sizes': self.filter_sizes
        })
        return config


class ForwardDoubleConnectedDecoder(Layer):

    def __init__(self, connections1: List[keras_layer.Layer], connections2: List[keras_layer.Layer], filter_sizes: Tuple[int] = (256, 128, 64, 32),
                 name: str = None, **kwargs):
        super(ForwardDoubleConnectedDecoder, self).__init__(name=name, **kwargs)

        self.filter_sizes = filter_sizes

        self.connections1 = connections1
        self.connections2 = connections2

        self.upsamplings: List[UpSampling2D] = []
        self.concatenations: List[concatenate] = []
        self.convolutions: List[ConvolutionalBlock] = []

        for index, filter_size in enumerate(self.filter_sizes):
            upsampling = UpSampling2D((2, 2), interpolation='bilinear')
            self.upsamplings.append(upsampling)

            concatenation = concatenate
            self.concatenations.append(concatenation)

            conv_block = ConvolutionalBlock(filters=filter_size)
            self.convolutions.append(conv_block)

    def call(self, inputs, **kwargs):
        skip_connections1 = self.connections1.copy()
        skip_connections1.reverse()

        skip_connections2 = self.connections2.copy()
        skip_connections2.reverse()

        x = inputs

        for index, filter_size in enumerate(self.filter_sizes):
            x = self.upsamplings[index](x)
            x = self.concatenations[index]([x, skip_connections1[index], skip_connections2[index]])
            x = self.convolutions[index](x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections1': self.__connections1,
            'connections2': self.__connections2,
            'filter_sizes': self.__filter_sizes
        })
        return config


class Squeeze_excite_block(Layer):
    def __init__(self, filters: int, ratio: int = 8):
        super(Squeeze_excite_block, self).__init__()
        self.__filters = filters
        self.__ratio = ratio

        self.gap = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, filters))
        self.dense1 = Dense(units=filters//ratio,
                            activation='relu',
                            kernel_initializer='he_normal',
                            use_bias=False)
        self.dense2 = Dense(units=filters,
                            activation='sigmoid',
                            kernel_initializer='he_normal',
                            use_bias=False)

    def call(self, inputs, **kwargs):
        init = inputs
        x = self.gap(init)
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = multiply([init, x])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'ratio': self.__ratio,
        })
        return config


class ConvolutionalBlock(Layer):
    def __init__(self, filters):
        super(ConvolutionalBlock, self).__init__()
        self.__filters = filters

        self.conv_block1 = ConvBlock(filters=filters,
                                     kSize=3)
        self.conv_block2 = ConvBlock(filters=filters,
                                     kSize=3)
        self.squeeze = Squeeze_excite_block(filters=filters)

    def call(self, inputs, **kwargs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.squeeze(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters
        })
        return config


class OutputBlock(Layer):

    def __init__(self, name: str = None, **kwargs):
        super(OutputBlock, self).__init__(name=name, **kwargs)

        self.conv2d = Conv2D(1, (1, 1), padding="same")
        self.activation = sigmoid

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv2d(x)
        x = self.activation(x)
        return x

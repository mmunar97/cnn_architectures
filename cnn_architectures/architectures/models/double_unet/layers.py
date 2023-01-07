import tensorflow.keras.layers as keras_layer

from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from typing import List


class VGGEncoder(Layer):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        self.__vgg_model = None
        self.__skipped_connections = []

    def call(self, inputs, *args, **kwargs):
        x = inputs
        self.__vgg_model = VGG19(include_top=False, weights='imagenet', input_tensor=x)
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for name in names:
            self.__skipped_connections.append(self.__vgg_model.get_layer(name).output)

        output = self.__vgg_model.get_layer("block5_conv4").output
        return output, self.__skipped_connections


class AtrousSpatialPyramidPooling(Layer):

    def __init__(self, n_filters: int):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.__n_filter = n_filters
        self.__avg_pooling_2d = None
        self.__conv2d_1 = None
        self.__batch_normalization_1 = None
        self.__activation_1 = None
        self.__upsampling_2d = None

        self.__conv2d_2 = None
        self.__batch_normalization_2 = None
        self.__activation_2 = None

        self.__conv2d_3 = None
        self.__batch_normalization_3 = None
        self.__activation_3 = None

        self.__conv2d_4 = None
        self.__batch_normalization_4 = None
        self.__activation_4 = None

        self.__conv2d_5 = None
        self.__batch_normalization_5 = None
        self.__activation_5 = None

        self.__concatenate = None

        self.__conv2d_6 = None
        self.__batch_normalization_6 = None
        self.__activation_6 = None

    def call(self, inputs, *args, **kwargs):
        shape = inputs.shape
        x = inputs

        self.__avg_pooling_2d = AveragePooling2D(pool_size=(shape[1], shape[2]))
        y1 = self.__avg_pooling_2d(x)
        self.__conv2d_1 = Conv2D(self.__n_filter, 1, padding="same")
        y1 = self.__conv2d_1(y1)
        self.__batch_normalization_1 = BatchNormalization()
        y1 = self.__batch_normalization_1(y1)
        self.__activation_1 = Activation("relu")
        y1 = self.__activation_1(y1)
        self.__upsampling_2d = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')
        y1 = self.__upsampling_2d(y1)

        self.__conv2d_2 = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)
        y2 = self.__conv2d_2(x)
        self.__batch_normalization_2 = BatchNormalization()
        y2 = self.__batch_normalization_2(y2)
        self.__activation_2 = Activation("relu")
        y2 = self.__activation_2(y2)

        self.__conv2d_3 = Conv2D(self.__n_filter, 3, dilation_rate=6, padding="same", use_bias=False)
        y3 = self.__conv2d_3(x)
        self.__batch_normalization_3 = BatchNormalization()
        y3 = self.__batch_normalization_3(y3)
        self.__activation_3 = Activation("relu")
        y3 = self.__activation_3(y3)

        self.__conv2d_4 = Conv2D(self.__n_filter, 3, dilation_rate=12, padding="same", use_bias=False)
        y4 = self.__conv2d_4(x)
        self.__batch_normalization_4 = BatchNormalization()
        y4 = self.__batch_normalization_4(y4)
        self.__activation_4 = Activation("relu")
        y4 = self.__activation_4(y4)

        self.__conv2d_5 = Conv2D(self.__n_filter, 3, dilation_rate=18, padding="same", use_bias=False)
        y5 = self.__conv2d_5(x)
        self.__batch_normalization_5 = BatchNormalization()
        y5 = self.__batch_normalization_5(y5)
        self.__activation_5 = Activation("relu")
        y5 = self.__activation_5(y5)

        self.__concatenate = Concatenate()
        y = self.__concatenate([y1, y2, y3, y4, y5])

        self.__conv2d_6 = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)
        y = self.__conv2d_6(y)
        self.__batch_normalization_6 = BatchNormalization()
        y = self.__batch_normalization_6(y)
        self.__activation_6 = Activation("relu")
        y = self.__activation_6(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters': self.__n_filter
        })
        return config


class ForwardConnectedDecoder(Layer):

    def __init__(self, connections: List[keras_layer.Layer]):
        super(ForwardConnectedDecoder, self).__init__()

        self.__connections = connections
        self.__upsamplings: List[UpSampling2D] = []
        self.__concatenations: List[Concatenate] = []
        self.__convolutions: List[ConvolutionalBlock] = []

    def call(self, inputs, **kwargs):
        num_filters = [256, 128, 64, 32]

        skip_connections = self.__connections.copy()
        skip_connections.reverse()
        x = inputs

        for i, f in enumerate(num_filters):
            upsampling2d = UpSampling2D((2, 2), interpolation='bilinear')
            self.__upsamplings.append(upsampling2d)
            x = upsampling2d(x)

            concat = Concatenate()
            self.__concatenations.append(concat)
            x = concat([x, skip_connections[i]])

            conv = ConvolutionalBlock(n_filters=f)
            self.__convolutions.append(conv)
            x = conv(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections': self.__connections
        })
        return config


class ForwardEncoder(Layer):

    def __init__(self):
        super(ForwardEncoder, self).__init__()

        self.__convolutions: List[ConvolutionalBlock] = []
        self.__pools: List[MaxPool2D] = []
        self.__skipped_connections = []

    def call(self, inputs, **kwargs):
        num_filters = [32, 64, 128, 256]
        x = inputs

        for i, f in enumerate(num_filters):
            conv = ConvolutionalBlock(n_filters=f)
            self.__convolutions.append(conv)
            x = conv(x)
            self.__skipped_connections.append(x)

            max_pool = MaxPool2D((2, 2))
            self.__pools.append(max_pool)
            x = max_pool(x)

        return x, self.__skipped_connections


class ForwardDoubleConnectedDecoder(Layer):

    def __init__(self, connections1: List[keras_layer.Layer], connections2: List[keras_layer.Layer]):
        super(ForwardDoubleConnectedDecoder, self).__init__()

        self.__connections1 = connections1
        self.__connections2 = connections2

        self.__upsamplings: List[UpSampling2D] = []
        self.__concatenations: List[Concatenate] = []
        self.__convolutions: List[ConvolutionalBlock] = []

    def call(self, inputs, **kwargs):
        num_filters = [256, 128, 64, 32]

        skip_connections1 = self.__connections1.copy()
        skip_connections1.reverse()

        skip_connections2 = self.__connections2.copy()
        skip_connections2.reverse()

        x = inputs

        for i, f in enumerate(num_filters):
            upsampling = UpSampling2D((2, 2), interpolation='bilinear')
            self.__upsamplings.append(upsampling)
            x = upsampling(x)

            concatenation = Concatenate()
            self.__concatenations.append(concatenation)
            x = concatenation([x, skip_connections1[i], skip_connections2[i]])

            conv_block = ConvolutionalBlock(n_filters=f)
            self.__convolutions.append(conv_block)
            x = conv_block(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections1': self.__connections1,
            'connections2': self.__connections2
        })
        return config


class ConvolutionalBlock(Layer):

    def __init__(self, n_filters: int):
        super(ConvolutionalBlock, self).__init__()

        self.__n_filter = n_filters

        self.__conv2d_1 = None
        self.__batch_normalization_1 = None
        self.__activation_1 = None
        self.__conv2d_2 = None
        self.__batch_normalization_2 = None
        self.__activation_2 = None
        self.__global_avg_pool_2d = None
        self.__reshape = None
        self.__dense1 = None
        self.__dense2 = None
        self.__mult = None

    def call(self, inputs, **kwargs):
        x = inputs

        self.__conv2d_1 = Conv2D(self.__n_filter, (3, 3), padding="same")
        x = self.__conv2d_1(x)
        self.__batch_normalization_1 = BatchNormalization()
        x = self.__batch_normalization_1(x)
        self.__activation_1 = Activation('relu')
        x = self.__activation_1(x)

        self.__conv2d_2 = Conv2D(self.__n_filter, (3, 3), padding="same")
        x = self.__conv2d_2(x)
        self.__batch_normalization_2 = BatchNormalization()
        x = self.__batch_normalization_2(x)
        self.__activation_2 = Activation('relu')
        x = self.__activation_2(x)

        x = self.__squeeze_excite_block(x)

        return x

    def __squeeze_excite_block(self, input_layer: keras_layer.Layer, ratio: int = 8):
        x = input_layer
        channel_axis = -1
        filters = x.shape[channel_axis]
        se_shape = (1, 1, filters)

        self.__global_avg_pool_2d = GlobalAveragePooling2D()
        se = self.__global_avg_pool_2d(x)
        self.__reshape = Reshape(se_shape)
        se = self.__reshape(se)
        self.__dense1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
        se = self.__dense1(se)
        self.__dense2 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        se = self.__dense2(se)

        self.__mult = Multiply()
        x = self.__mult([x, se])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters': self.__n_filter
        })
        return config


class OutputBlock(Layer):

    def __init__(self):
        super(OutputBlock, self).__init__()

        self.__conv2d = Conv2D(1, (1, 1), padding="same")
        self.__activation = Activation('sigmoid')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.__conv2d(x)
        x = self.__activation(x)
        return x

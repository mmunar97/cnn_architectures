import tensorflow as tf
import tensorflow.keras.layers as keras_layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from typing import List


class VGGEncoder(Layer):
    def __init__(self, vgg_model: Model, name: str = None, **kwargs):
        super(VGGEncoder, self).__init__(name=name, **kwargs)

        self.__vgg_model = vgg_model
        self.__skipped_connections = []

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, *args, **kwargs):
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for name in names:
            self.__skipped_connections.append(self.__vgg_model.get_layer(name).output)

        output = self.__vgg_model.get_layer("block5_conv4").output
        return output, self.__skipped_connections

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vgg_model': self.__vgg_model
        })
        return config


class AtrousSpatialPyramidPooling(Layer):

    def __init__(self, n_filters: int, name: str = None, **kwargs):
        super(AtrousSpatialPyramidPooling, self).__init__(name=name, **kwargs)

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

    def build(self, input_shape):
        self.__avg_pooling_2d = AveragePooling2D(pool_size=(input_shape[1], input_shape[2]))
        self.__conv2d_1 = Conv2D(self.__n_filter, 1, padding="same")
        self.__batch_normalization_1 = BatchNormalization()
        self.__activation_1 = Activation("relu")
        self.__upsampling_2d = UpSampling2D((input_shape[1], input_shape[2]), interpolation='bilinear')
        self.__conv2d_2 = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)
        self.__batch_normalization_2 = BatchNormalization()
        self.__activation_2 = Activation("relu")
        self.__conv2d_3 = Conv2D(self.__n_filter, 3, dilation_rate=6, padding="same", use_bias=False)
        self.__batch_normalization_3 = BatchNormalization()
        self.__activation_3 = Activation("relu")
        self.__conv2d_4 = Conv2D(self.__n_filter, 3, dilation_rate=12, padding="same", use_bias=False)
        self.__batch_normalization_4 = BatchNormalization()
        self.__activation_4 = Activation("relu")
        self.__conv2d_5 = Conv2D(self.__n_filter, 3, dilation_rate=18, padding="same", use_bias=False)
        self.__batch_normalization_5 = BatchNormalization()
        self.__activation_5 = Activation("relu")
        self.__concatenate = concatenate
        self.__conv2d_6 = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)
        self.__batch_normalization_6 = BatchNormalization()
        self.__activation_6 = Activation("relu")

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, *args, **kwargs):
        x = inputs

        y1 = self.__avg_pooling_2d(x)
        y1 = self.__conv2d_1(y1)
        y1 = self.__batch_normalization_1(y1)
        y1 = self.__activation_1(y1)
        y1 = self.__upsampling_2d(y1)

        y2 = self.__conv2d_2(x)
        y2 = self.__batch_normalization_2(y2)
        y2 = self.__activation_2(y2)

        y3 = self.__conv2d_3(x)
        y3 = self.__batch_normalization_3(y3)
        y3 = self.__activation_3(y3)

        y4 = self.__conv2d_4(x)
        y4 = self.__batch_normalization_4(y4)
        y4 = self.__activation_4(y4)

        y5 = self.__conv2d_5(x)
        y5 = self.__batch_normalization_5(y5)
        y5 = self.__activation_5(y5)

        y = self.__concatenate([y1, y2, y3, y4, y5])
        y = self.__conv2d_6(y)
        y = self.__batch_normalization_6(y)
        y = self.__activation_6(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters': self.__n_filter
        })
        return config


class ForwardConnectedDecoder(Layer):

    def __init__(self, connections: List[keras_layer.Layer], filter_sizes: List[int] = None, name: str = None, **kwargs):
        super(ForwardConnectedDecoder, self).__init__(name=name, **kwargs)

        if filter_sizes is None:
            self.__filter_sizes = [256, 128, 64, 32]
        else:
            self.__filter_sizes = filter_sizes

        self.__connections = connections
        self.__upsamplings: List[UpSampling2D] = []
        self.__concatenations: List[concatenate] = []
        self.__convolutions: List[ConvolutionalBlock] = []

    def build(self, input_shape):
        for index, filter_size in enumerate(self.__filter_sizes):
            upsampling2d = UpSampling2D((2, 2), interpolation='bilinear')
            self.__upsamplings.append(upsampling2d)

            concat = concatenate
            self.__concatenations.append(concat)

            conv = ConvolutionalBlock(n_filters=filter_size)
            self.__convolutions.append(conv)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):

        skip_connections = self.__connections.copy()
        skip_connections.reverse()
        x = inputs

        for index, filter_size in enumerate(self.__filter_sizes):
            x = self.__upsamplings[index](x)
            x = self.__concatenations[index]([x, skip_connections[index]])
            x = self.__convolutions[index](x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections': self.__connections,
            'filter_sizes': self.__filter_sizes
        })
        return config


class ForwardEncoder(Layer):

    def __init__(self, filter_sizes: List[int] = None, name: str = None, **kwargs):
        super(ForwardEncoder, self).__init__(name=name, **kwargs)

        if filter_sizes is None:
            self.__filter_sizes = [32, 64, 128, 256]
        else:
            self.__filter_sizes = filter_sizes

        self.__convolutions: List[ConvolutionalBlock] = []
        self.__pools: List[MaxPool2D] = []
        self.__skipped_connections = []

    def build(self, input_shape):
        for index, filter_size in enumerate(self.__filter_sizes):
            conv = ConvolutionalBlock(n_filters=filter_size)
            self.__convolutions.append(conv)

            max_pool = MaxPool2D((2, 2))
            self.__pools.append(max_pool)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        x = inputs

        for index, filter_size in enumerate(self.__filter_sizes):
            x = self.__convolutions[index](x)
            self.__skipped_connections.append(x)
            x = self.__pools[index](x)

        return x, self.__skipped_connections

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_sizes': self.__filter_sizes
        })
        return config


class ForwardDoubleConnectedDecoder(Layer):

    def __init__(self, connections1: List[keras_layer.Layer], connections2: List[keras_layer.Layer], filter_sizes: List[int] = None,
                 name: str = None, **kwargs):
        super(ForwardDoubleConnectedDecoder, self).__init__(name=name, **kwargs)

        if filter_sizes is None:
            self.__filter_sizes = [256, 128, 64, 32]
        else:
            self.__filter_sizes = filter_sizes

        self.__connections1 = connections1
        self.__connections2 = connections2

        self.__upsamplings: List[UpSampling2D] = []
        self.__concatenations: List[concatenate] = []
        self.__convolutions: List[ConvolutionalBlock] = []

    def build(self, input_shape):
        for index, filter_size in enumerate(self.__filter_sizes):
            upsampling = UpSampling2D((2, 2), interpolation='bilinear')
            self.__upsamplings.append(upsampling)

            concatenation = concatenate
            self.__concatenations.append(concatenation)

            conv_block = ConvolutionalBlock(n_filters=filter_size)
            self.__convolutions.append(conv_block)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        skip_connections1 = self.__connections1.copy()
        skip_connections1.reverse()

        skip_connections2 = self.__connections2.copy()
        skip_connections2.reverse()

        x = inputs

        for index, filter_size in enumerate(self.__filter_sizes):
            x = self.__upsamplings[index](x)
            x = self.__concatenations[index]([x, skip_connections1[index], skip_connections2[index]])
            x = self.__convolutions[index](x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'connections1': self.__connections1,
            'connections2': self.__connections2,
            'filter_sizes': self.__filter_sizes
        })
        return config


class ConvolutionalBlock(Layer):

    def __init__(self, n_filters: int, squeeze_ratio: int = 8, name: str = None, **kwargs):
        super(ConvolutionalBlock, self).__init__(name=name, **kwargs)

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
        self.__squeeze_ratio = squeeze_ratio

    def build(self, input_shape):
        self.__conv2d_1 = Conv2D(self.__n_filter, (3, 3), padding="same")
        self.__batch_normalization_1 = BatchNormalization()
        self.__activation_1 = Activation('relu')
        self.__conv2d_2 = Conv2D(self.__n_filter, (3, 3), padding="same")
        self.__batch_normalization_2 = BatchNormalization()
        self.__activation_2 = Activation('relu')
        self.__global_avg_pool_2d = GlobalAveragePooling2D()

        previous_layers = [self.__conv2d_1, self.__batch_normalization_1, self.__activation_1,
                           self.__conv2d_2, self.__batch_normalization_2, self.__activation_2]

        output_shape = input_shape
        for layer in previous_layers:
            output_shape = layer.compute_output_shape(output_shape)

        self.__reshape = Reshape((1, 1, output_shape[-1]))
        self.__dense1 = Dense(output_shape[-1] // self.__squeeze_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.__dense2 = Dense(output_shape[-1], activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.__mult = Multiply()

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        x = inputs

        x = self.__conv2d_1(x)
        x = self.__batch_normalization_1(x)
        x = self.__activation_1(x)
        x = self.__conv2d_2(x)
        x = self.__batch_normalization_2(x)
        x = self.__activation_2(x)

        # Squeeze-and-excite block
        se = self.__global_avg_pool_2d(x)
        se = self.__reshape(se)
        se = self.__dense1(se)
        se = self.__dense2(se)

        x = self.__mult([x, se])

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters': self.__n_filter,
            'squeeze_ratio': self.__squeeze_ratio
        })
        return config


class OutputBlock(Layer):

    def __init__(self, name: str = None, **kwargs):
        super(OutputBlock, self).__init__(name=name, **kwargs)

        self.__conv2d = Conv2D(1, (1, 1), padding="same")
        self.__activation = Activation('sigmoid')

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        x = inputs
        x = self.__conv2d(x)
        x = self.__activation(x)
        return x

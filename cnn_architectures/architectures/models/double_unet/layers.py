import tensorflow.keras.layers as keras_layer

from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from typing import List


class VGGEncoder(Layer):
    def __init__(self):
        super(VGGEncoder, self).__init__()

    def call(self, inputs, *args, **kwargs):
        skip_connections = []

        model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for name in names:
            skip_connections.append(model.get_layer(name).output)

        output = model.get_layer("block5_conv4").output
        return output, skip_connections


class AtrousSpatialPyramidPooling(Layer):

    def __init__(self, n_filters: int):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.__n_filter = n_filters

    def call(self, inputs, *args, **kwargs):
        shape = inputs.shape
        x = inputs

        y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
        y1 = Conv2D(self.__n_filter, 1, padding="same")(y1)
        y1 = BatchNormalization()(y1)
        y1 = Activation("relu")(y1)
        y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

        y2 = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
        y2 = BatchNormalization()(y2)
        y2 = Activation("relu")(y2)

        y3 = Conv2D(self.__n_filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
        y3 = BatchNormalization()(y3)
        y3 = Activation("relu")(y3)

        y4 = Conv2D(self.__n_filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
        y4 = BatchNormalization()(y4)
        y4 = Activation("relu")(y4)

        y5 = Conv2D(self.__n_filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
        y5 = BatchNormalization()(y5)
        y5 = Activation("relu")(y5)

        y = Concatenate()([y1, y2, y3, y4, y5])

        y = Conv2D(self.__n_filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

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

    def call(self, inputs, **kwargs):
        num_filters = [256, 128, 64, 32]

        skip_connections = self.__connections.copy()
        skip_connections.reverse()
        x = inputs

        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = Concatenate()([x, skip_connections[i]])

            x = ConvolutionalBlock(n_filters=f)(x)

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

    def call(self, inputs, **kwargs):
        num_filters = [32, 64, 128, 256]
        skip_connections = []
        x = inputs

        for i, f in enumerate(num_filters):
            x = ConvolutionalBlock(n_filters=f)(x)

            skip_connections.append(x)
            x = MaxPool2D((2, 2))(x)

        return x, skip_connections


class ForwardDoubleConnectedDecoder(Layer):

    def __init__(self, connections1: List[keras_layer.Layer], connections2: List[keras_layer.Layer]):
        super(ForwardDoubleConnectedDecoder, self).__init__()

        self.__connections1 = connections1
        self.__connections2 = connections2

    def call(self, inputs, **kwargs):
        num_filters = [256, 128, 64, 32]

        skip_connections1 = self.__connections1.copy()
        skip_connections1.reverse()

        skip_connections2 = self.__connections2.copy()
        skip_connections2.reverse()

        x = inputs

        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = Concatenate()([x, skip_connections1[i], skip_connections2[i]])

            x = ConvolutionalBlock(n_filters=f)(x)

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

    def call(self, inputs, **kwargs):
        x = inputs

        x = Conv2D(self.__n_filter, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(self.__n_filter, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = self.__squeeze_excite_block(x)

        return x

    @staticmethod
    def __squeeze_excite_block(input_layer: keras_layer.Layer, ratio: int = 8):
        init = input_layer
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = Multiply()([init, se])
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

    def call(self, inputs, **kwargs):
        x = Conv2D(1, (1, 1), padding="same")(inputs)
        x = Activation('sigmoid')(x)
        return x

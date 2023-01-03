import tensorflow.keras.layers as keras_layer

from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from typing import List


class VGGEncoder:

    def __init__(self, input_layer: keras_layer.Input):
        self.__input = input_layer

    def build_layer(self):
        skip_connections = []

        model = VGG19(include_top=False, weights='imagenet', input_tensor=self.__input)
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for name in names:
            skip_connections.append(model.get_layer(name).output)

        output = model.get_layer("block5_conv4").output
        return output, skip_connections


class AtrousSpatialPyramidPooling:

    def __init__(self, input_layer: keras_layer.Layer, n_filters: int):
        self.__input = input_layer
        self.__n_filter = n_filters

    def build_layer(self):
        shape = self.__input.shape
        x = self.__input

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


class ForwardConnectedDecoder:

    def __init__(self, input_layer: keras_layer.Layer, connections: List[keras_layer.Layer]):
        self.__input = input_layer
        self.__connections = connections

    def build_layer(self):
        num_filters = [256, 128, 64, 32]

        skip_connections = self.__connections.copy()
        skip_connections.reverse()
        x = self.__input

        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = Concatenate()([x, skip_connections[i]])

            convolutional_block = ConvolutionalBlock(input_layer=x, n_filters=f)
            x = convolutional_block.build_layer()

        return x


class ForwardEncoder:

    def __init__(self, input_layer: keras_layer.Layer):
        self.__input = input_layer

    def build_layer(self):
        num_filters = [32, 64, 128, 256]
        skip_connections = []
        x = self.__input

        for i, f in enumerate(num_filters):
            convolutional_block = ConvolutionalBlock(input_layer=x, n_filters=f)
            x = convolutional_block.build_layer()

            skip_connections.append(x)
            x = MaxPool2D((2, 2))(x)

        return x, skip_connections


class ForwardDoubleConnectedDecoder:

    def __init__(self, input_layer: keras_layer.Layer, connections1: List[keras_layer.Layer], connections2: List[keras_layer.Layer]):
        self.__input = input_layer
        self.__connections1 = connections1
        self.__connections2 = connections2

    def build_layer(self):
        num_filters = [256, 128, 64, 32]

        skip_connections1 = self.__connections1.copy()
        skip_connections1.reverse()

        skip_connections2 = self.__connections2.copy()
        skip_connections2.reverse()

        x = self.__input

        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = Concatenate()([x, skip_connections1[i], skip_connections2[i]])

            convolutional_block = ConvolutionalBlock(input_layer=x, n_filters=f)
            x = convolutional_block.build_layer()

        return x


class ConvolutionalBlock:

    def __init__(self, input_layer: keras_layer.Layer, n_filters: int):
        self.__input = input_layer
        self.__n_filter = n_filters

    def build_layer(self):
        x = self.__input

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


class OutputBlock:

    def __init__(self, input_layer: keras_layer.Layer):
        self.__input = input_layer

    def build_layer(self):
        x = Conv2D(1, (1, 1), padding="same")(self.__input)
        x = Activation('sigmoid')(x)
        return x

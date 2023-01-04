from typing import Tuple

import tensorflow as tf
import tensorflow.keras.layers as keras_layer


class ConvBlock(keras_layer.Layer):
    """
    Convolutional block used on the encoder
    """

    def __init__(self, layer_idx: int, filters: int, kernel_size: Tuple[int, int],
                 activation: str, kernel_initializer: str = 'he_normal', padding: str = "same",
                 batch_normalization: bool = False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.__layer_idx: int = layer_idx
        self.__filters: int = filters
        self.__kernel_size = kernel_size
        self.__activation: str = activation
        self.__kernel_initializer: str = kernel_initializer
        self.__padding: str = padding
        self.__is_batch_normalized: bool = batch_normalization

        self.conv2d_1 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)

        if batch_normalization:
            self.batch_normalization_1 = keras_layer.BatchNormalization()

        self.conv2d_2 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)
        self.batch_normalization_2 = keras_layer.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_idx': self.__layer_idx,
            'filters': self.__filters,
            'kernel_size': self.__kernel_size,
            'activation': self.__activation,
            'kernel_initializer': self.__kernel_initializer,
            'padding': self.__padding,
            'batch_normalization': self.__is_batch_normalized
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if self.__is_batch_normalized:
            x = self.batch_normalization_1(x)

        x = self.conv2d_2(x)

        if self.__is_batch_normalized:
            x = self.batch_normalization_2(x)

        return x

    @property
    def layer_idx(self):
        return self.__layer_idx


class UpConvBlock(keras_layer.Layer):
    """
    Block to build the decoder. The decoder is build with the combination of UpSampling2D and Conv2D.
    """

    def __init__(self, layer_idx: int, filter_size: Tuple[int, int], filters: int,
                 activation: str = 'relu', padding: str = "same",
                 kernel_initializer: str = "he_normal", **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)

        self.__layer_idx = layer_idx
        self.__filter_size = filter_size
        self.__filters = filters
        self.__activation = activation
        self.__padding = padding
        self.__kernel_initializer = kernel_initializer

        self.up_sampling_1 = keras_layer.UpSampling2D(size=filter_size)
        self.conv2d_1 = keras_layer.Conv2D(filters, kernel_size=filter_size,
                                           activation=activation, padding=padding,
                                           kernel_initializer=kernel_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.up_sampling_1(x)
        x = self.conv2d_1(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_idx': self.__layer_idx,
            'filter_size': self.__filter_size,
            'filters': self.__filters,
            'activation': self.__activation,
            'padding': self.__padding,
            'kernel_initializer': self.__kernel_initializer,
        })
        return config


class CropConcatBlock(keras_layer.Layer):

    @tf.autograph.experimental.do_not_convert
    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff), :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x
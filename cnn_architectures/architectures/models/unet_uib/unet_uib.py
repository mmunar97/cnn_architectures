from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.unet.layers import *
from tensorflow.keras.optimizers import *
from typing import Callable, Union

import tensorflow.keras.models as keras_model
import warnings


class UNetUIB(CNNModel):
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]], out_channel: int,
                 batch_normalization: bool):
        super().__init__(input_size)

        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel

        self.__history = None

    def build(self, n_filters, last_activation: Union[Callable, str], dilation_rate: int = 1,
              layer_depth: int = 5, kernel_size: Tuple[int, int] = (3, 3),
              pool_size: Tuple[int, int] = (2, 2)):
        """
        Builds the graph and model for the modified U-Net. This model has the ability to perform
        binary classification by terminating the encoder structure.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            last_activation:
            dilation_rate:
            layer_depth:
            kernel_size:
            pool_size:

        """
        # Define input batch shape
        input_image = keras_layer.Input(self.input_size, name="input_image")
        encoder = {}

        conv_params = dict(filters=n_filters,
                           kernel_size=kernel_size,
                           activation='relu',
                           batch_normalization=True)

        x = input_image
        layer_idx = 0

        for layer_idx in range(0, layer_depth):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = ConvBlock(layer_idx, **conv_params)(x)
            encoder[layer_idx] = x

            x = keras_layer.MaxPooling2D(pool_size)(x)

        y = keras_layer.Flatten()(x)
        y = keras_layer.Dense(1024, activation='relu')(y)
        y = keras_layer.Dense(1024, activation='relu')(y)
        y = keras_layer.Dense(2, activation='softmax', name="class_out")(y)

        for layer_idx in range(layer_idx, -1, -1):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = UpConvBlock(layer_idx, filter_size=(2, 2), filters=n_filters * (2 ** layer_idx),
                            activation='relu')(x)
            x = CropConcatBlock()(x, encoder[layer_idx])
            x = ConvBlock(layer_idx, **conv_params)(x)

        mask_out = keras_layer.Conv2D(self.__n_channels, (1, 1), activation=last_activation,
                                      padding='same', dilation_rate=dilation_rate,
                                      kernel_initializer='he_normal', name="img_out")(x)

        model = keras_model.Model(inputs=input_image, outputs=[mask_out, y])

        self.set_model(model)

        return input_image, encoder, mask_out
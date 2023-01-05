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

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy",
                learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """ Compiles the model.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            learning_rate (Num). Learning rate of the training

        Returns:

        """
        # loss_functions = {"img_out": loss_func}

        self.model.compile(*args, **kwargs, optimizer=Adam(learning_rate=learning_rate),
                           loss=loss_func, metrics=['categorical_accuracy', 'binary_accuracy'])

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None, verbose=1,
              *args, **kwargs):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. The info is defined
        on Config class or instead passed as parameters.

        Args:
            train_generator:
            val_generator:
            epochs:
            steps_per_epoch:
            validation_steps:
            check_point_path:
            callbacks:
            verbose:

        Returns:

        """
        if self.__history is not None:
            warnings.warn("Model already trained, starting new training")

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                                save_weights_only=False,
                                                                save_best_only=True))

        if val_generator is not None:
            history = self.model.fit(train_generator, validation_data=val_generator,
                                     epochs=epochs,
                                     validation_steps=validation_steps,
                                     callbacks=callbacks,
                                     steps_per_epoch=steps_per_epoch,
                                     verbose=verbose, *args, **kwargs)
        else:
            history = self.model.fit(train_generator, epochs=epochs,
                                     callbacks=callbacks, verbose=verbose,
                                     steps_per_epoch=steps_per_epoch, *args,
                                     **kwargs)

        self.__history = history

    def load_weight(self, path: str):
        self.model.load_weights(path)

    @property
    def history(self):
        return self.__history

    def get_layer(self, *args, **kwargs):
        """ Wrapper of the Keras get_layer function.
        """
        return self.model.get_layer(*args, **kwargs)

    def summary(self):
        self.model.summary()

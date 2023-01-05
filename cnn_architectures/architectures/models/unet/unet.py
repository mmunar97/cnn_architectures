from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.unet.layers import *
from tensorflow.keras.optimizers import *
from typing import Callable, Union, List

import tensorflow.keras.models as keras_model
import warnings


class UNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 out_channel: int,
                 batch_normalization: bool):
        """
        Initializes the model which represents the GSC architecture.

        References:
            U-Net: Convolutional Networks for Biomedical Image Segmentation. O Ronneberger, P Fisher, T Brox.
            Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015

        Args:
            input_size: A tuple of three elements or two elements, depending on the input images, respectively.
            out_channel: An integer, representing the number of output channels.
            batch_normalization: A boolean, representing if batch normalization has to be applied.
        """
        super().__init__(input_size)

        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel

        self.__history = None

    def build(self, n_filters: int, last_activation: Union[Callable, str], dilation_rate: int = 1,
              layer_depth: int = 5, kernel_size: Tuple[int, int] = (3, 3),
              pool_size: Tuple[int, int] = (2, 2)):
        """
        Builds the model and constructs the graph.
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

        for layer_idx in range(layer_idx, -1, -1):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = UpConvBlock(layer_idx, filter_size=(2, 2), filters=n_filters * (2 ** layer_idx),
                            activation='relu')(x)
            x = CropConcatBlock()(x, encoder[layer_idx])
            x = ConvBlock(layer_idx, **conv_params)(x)

        mask_out = keras_layer.Conv2D(self.__n_channels, (1, 1), activation=last_activation,
                                      padding='same', dilation_rate=dilation_rate,
                                      kernel_initializer='he_normal', name="img_out")(x)

        model = keras_model.Model(inputs=input_image, outputs=mask_out)

        self.set_model(model)

        return input_image, encoder, mask_out

    def compile(self,
                loss_func: Union[str, Callable] = "categorical_crossentropy",
                metrics: List[Union[str, Callable]] = ["accuracy"],
                learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """
        Compiles the model.

        Args:
            loss_func: An string or callable method, which represent the loss function to be used in the training.
            metrics: A list of strings or callable methods, which represent the metrics to measure the training performance.
            learning_rate: An integer or a float number, representing the learning rate of the training.
        """
        loss_functions = {"img_out": loss_func}

        self.model.compile(*args, **kwargs, optimizer=Adam(learning_rate=learning_rate),
                           loss=loss_functions, metrics=metrics)

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None, verbose=1,
              *args, **kwargs):
        """
        Trains the model with the info passed as parameters.

        Args:
            train_generator: A generator, representing the feeder for the training process.
            val_generator: A generator, representing the feeder for the validation process while training.
            epochs: An integer, representing the number of epochs to use.
            steps_per_epoch: An integer, representing the number of steps to perform in each epoch.
            validation_steps: An integer, representing the number of steps in the validation.
            check_point_path: A string, representing the path where the checkpoints will be saved. Can be None.
            callbacks: A list of callable methods, representing the callbacks during the training.
            verbose: An integer, representing if the log has to be shown.
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
        return self.model

    def get_layer(self, *args, **kwargs):
        """
        Wrapper of the Keras get_layer function.
        """
        return self.model.get_layer(*args, **kwargs)

    def summary(self):
        self.model.summary()

from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from typing import Callable, Union, Tuple

from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.double_unet.layers import *
from cnn_architectures.utils.metrics import *

import tensorflow.keras.models as keras_model
import warnings


class DoubleUNet(CNNModel):

    def __init__(self, input_size: Tuple[int, int, int]):
        super().__init__(input_size)

        self.__history = None

    def build(self):
        input_image = keras_layer.Input(self.__input_size, name="input_image")

        encoder1 = VGGEncoder(input_layer=input_image)
        x, skip1 = encoder1.build_layer()

        asap1 = AtrousSpatialPyramidPooling(input_layer=x, n_filters=64)
        x = asap1.build_layer()

        decoder1 = ForwardConnectedDecoder(input_layer=x, connections=skip1)
        x = decoder1.build_layer()

        output1 = OutputBlock(input_layer=x)
        output1_layer = output1.build_layer()
        x = input_image * output1_layer

        encoder2 = ForwardEncoder(input_layer=x)
        x, skip2 = encoder2.build_layer()

        asap2 = AtrousSpatialPyramidPooling(input_layer=x, n_filters=64)
        x = asap2.build_layer()

        decoder2 = ForwardDoubleConnectedDecoder(input_layer=x, connections1=skip1, connections2=skip2)
        x = decoder2.build_layer()

        output2 = OutputBlock(input_layer=x)
        final_output = Concatenate()([output1_layer, output2.build_layer()])

        model = keras_model.Model(inputs=input_image, outputs=final_output)
        self.__internal_model = model

    def compile(self,
                metrics: List[Callable] = None,
                learning_rate: Union[int, float] = 1e-4,
                *args, **kwargs):
        """
        Compiles the model.
        """

        # Setting standard metrics for segmentation purposes
        if metrics is None:
            metrics = [dice_coef, iou, Recall(), Precision()]

        self.__internal_model.compile(*args, **kwargs, optimizer=Adam(learning_rate=learning_rate),
                                      metrics=metrics)

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None, verbose=1,
              *args, **kwargs):
        """
        Trains the model with the specified parameters.

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
            warnings.warn("Model already trained, starting new training.")

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                                save_weights_only=False,
                                                                save_best_only=True))

        if val_generator is not None:
            history = self.__internal_model.fit(train_generator, validation_data=val_generator,
                                                epochs=epochs,
                                                validation_steps=validation_steps,
                                                callbacks=callbacks,
                                                steps_per_epoch=steps_per_epoch,
                                                verbose=verbose, *args, **kwargs)
        else:
            history = self.__internal_model.fit(train_generator, epochs=epochs,
                                                callbacks=callbacks, verbose=verbose,
                                                steps_per_epoch=steps_per_epoch, *args,
                                                **kwargs)

        self.__history = history

    def load_weight(self, path: str):
        self.__internal_model.load_weights(path)

    @property
    def model(self):
        return self.__internal_model

    @property
    def history(self):
        return self.__history

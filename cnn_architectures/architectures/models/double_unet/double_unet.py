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
        input_image = keras_layer.Input(self.input_size, name="input_image")

        x, skip1 = VGGEncoder()(input_image)
        x = AtrousSpatialPyramidPooling(n_filters=64)(x)
        x = ForwardConnectedDecoder(connections=skip1)(x)

        output1 = OutputBlock()(x)
        x = input_image * output1

        x, skip2 = ForwardEncoder()(x)
        x = AtrousSpatialPyramidPooling(n_filters=64)(x)
        x = ForwardDoubleConnectedDecoder(input_layer=x, connections1=skip1, connections2=skip2)(x)

        output2 = OutputBlock()(x)
        final_output = Concatenate()([output1, output2])

        model = keras_model.Model(inputs=input_image, outputs=final_output)
        self.set_model(model)

    def compile(self,
                loss_func: List[Union[str, Callable]] = ["categorical_crossentropy"],
                metrics: List[Callable] = None,
                learning_rate: Union[int, float] = 1e-4,
                *args, **kwargs):
        """
        Compiles the model.
        """

        # Setting standard metrics for segmentation purposes
        if metrics is None:
            metrics = [dice_coef, iou, Recall(), Precision()]

        self.model.compile(*args, **kwargs, optimizer=Adam(learning_rate=learning_rate),
                           loss=loss_func, metrics=metrics)

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

import warnings

from tensorflow.keras.optimizers import *

from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.gsc.layers import EncoderBlock, ConvBlock, DecoderBlock
from tensorflow import keras
from typing import Tuple, List, Union, Callable


class GSC(CNNModel):

    def __init__(self,
                 input_size: Tuple[int, int] = (256, 256),
                 filters: List[int] = None):
        """
        Initializes the model which represents the GSC architecture.

        References:
            Promising crack segmentation method based on gated skip connection. M Jabreel and M Abdel-Nasser.
            Electronic Letters, Volume 56, Issue 10, 2020.

        Args:
            input_size: A tuple of two elements, representing the size of the input images. The GSC architecture only allows to handle grey-scale images.
            filters: A list of integers, representing the sizes of the consecutive filters to be applied. The first element of the list must be one,
                     since it is used for the final output.
        """
        super().__init__(input_size + (1,))

        if filters is None:
            filters = [1, 32, 64, 128, 256, 512, 1024]

        self.__filters = filters
        self.__history = None

    def build(self):
        """
        Builds the model and constructs the graph.
        """
        input_image = keras.layers.Input(self.input_size, name="input_image")

        y_history = []
        x = input_image

        for filter_size in self.__filters[1:len(self.__filters) - 1]:
            encoder = EncoderBlock(number_filters=filter_size)
            x, y_encoder = encoder(x)
            y_history.append(y_encoder)

        conv_block = ConvBlock(self.__filters[len(self.__filters) - 1])
        x = conv_block(x)
        y_history.reverse()

        for index, filter_size in enumerate(reversed(self.__filters[1:len(self.__filters) - 1])):
            decoder = DecoderBlock(number_filters=filter_size)
            x = decoder([x, y_history[index]])

        mask_out = keras.layers.Conv2D(filters=self.__filters[0], kernel_size=(1, 1), activation='sigmoid')(x)

        model = keras.models.Model(inputs=input_image, outputs=mask_out)
        self.set_model(model)

        return input_image, mask_out

    def compile(self,
                loss_func: List[Union[str, Callable]] = ["categorical_crossentropy"],
                metrics: List[Union[str, Callable]] = ["binary_accuracy"],
                learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """
        Compiles the model.

        Args:
            loss_func: A list of strings or callable methods, which represent the loss function to be used in the training.
            metrics: A list of strings or callable methods, which represent the metrics to measure the training performance.
            learning_rate: An integer or a float number, representing the learning rate of the training.
        """
        self.model.compile(*args, **kwargs, optimizer=Adam(learning_rate=learning_rate),
                           loss=loss_func, metrics=metrics)

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks: List[Callable] = None, verbose: int = 1,
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
            callbacks.append(keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
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
        """
        Loads the already-trained weights from a path.
        """
        self.model.load_weights(path)

    @property
    def history(self):
        return self.__history

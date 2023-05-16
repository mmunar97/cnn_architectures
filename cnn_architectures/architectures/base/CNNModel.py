from tensorflow import keras
from tensorflow.keras.optimizers import *
import cv2
import numpy
import skimage.transform
import tensorflow as tf
from typing import *
# tf.compat.v1.disable_eager_execution()


class CNNModel:

    def __init__(self,
                 input_size: Tuple[int, ...]):
        self.__input_size: Tuple[int, ...] = input_size
        self.__internal_model = None

    @property
    def input_size(self):
        return self.__input_size

    @property
    def model(self):
        return self.__internal_model

    def set_model(self, model: keras.models.Model):
        self.__internal_model = model

    def compile(self,
                optimizer: keras.optimizers = Adam,
                loss_func: Union[keras.losses.Loss, Callable, str] = 'categorical_crossentropy',
                metrics: List[Union[str, Callable]] = ['accuracy'],
                learning_rate: Union[int, float] = 3e-5,
                *args,
                **kwargs):

        loss_functions = {'mask_out': loss_func}

        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),
                           loss=loss_functions,
                           metrics=metrics, *args, **kwargs)

    def train(self,
              train_generator,
              epochs: int,
              steps_per_epoch: int,
              checkpoint_path: Union[str, None],
              checkpoint_freq: Union[int, str] = 'epoch',
              # save_all: bool=False,
              callbacks=None,
              verbose: int = 2,
              val_generator=None,
              val_steps: int = 0,
              *args, **kwargs,
              ):

        if callbacks is None:
            callbacks = []

        if checkpoint_path is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             verbose=0,
                                                             save_weights_only=False,
                                                             save_best_only=True if val_generator is not None else False,
                                                             save_freq=checkpoint_freq))

        if val_generator is not None:
            history = self.model.fit(train_generator, validation_data=val_generator,
                                     epochs=epochs,
                                     validation_steps=val_steps,
                                     callbacks=callbacks,
                                     steps_per_epoch=steps_per_epoch,
                                     verbose=verbose,
                                     *args, **kwargs)
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
        """
        Wrapper of the Keras get_layer function.
        """
        return self.model.get_layer(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def save(self, *args, **kwargs):
        self.model.optimizer = None
        self.model.compiled_loss = None
        self.model.compiled_metrics = None
        self.model.save(*args, **kwargs)

    def predict_binary(self, image: numpy.ndarray, binary_threshold: float,
                       prediction_index: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Performs the prediction of the already-trained model as a binary mask.

        Args:
            image: An image represented as a numpy array.
            binary_threshold: A float, representing the number to be used in the binarization.
            prediction_index: An integer, representing the position of the final output in the model prediction.

        Returns:
            Two numpy arrays: the first one, representing the binary mask, and the second one, representing the original image with the binary mask drawn over it.
        """
        if image.ndim == 2:
            resized_image_normalized = skimage.transform.resize(image, (self.__input_size[0], self.__input_size[1]))
        else:
            resized_image_normalized = skimage.transform.resize(image, (self.__input_size[0], self.__input_size[1], image.shape[2]))

        resized_image = cv2.resize(image, (self.__input_size[0], self.__input_size[1]))

        prediction = self.model.predict(numpy.array([resized_image_normalized]))
        prediction_mask_bool = prediction[0][:, :, prediction_index] >= binary_threshold
        prediction_mask_int = 255 * prediction_mask_bool

        if image.ndim == 2:
            resized_image[prediction_mask_bool] = 255

        else:
            resized_image[:, :, 0][prediction_mask_bool] = 0
            resized_image[:, :, 1][prediction_mask_bool] = 255
            resized_image[:, :, 2][prediction_mask_bool] = 0

        resized_image = cv2.resize(resized_image, (image.shape[1], image.shape[0]))

        return prediction_mask_bool, prediction_mask_int, resized_image

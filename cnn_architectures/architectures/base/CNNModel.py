from tensorflow import keras
from typing import Tuple

import cv2
import numpy
import skimage


class CNNModel:

    def __init__(self,
                 input_size: Tuple[int, ...]):
        self.__input_size: Tuple[int, ...] = input_size
        self.__internal_model: keras.models.Model = None

    def predict_binary(self, image: numpy.ndarray, binary_threshold: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Performs the prediction of the already-trained model as a binary mask.

        Args:
            image: An image represented as a numpy array.
            binary_threshold: A float, representing the number to be used in the binarization.

        Returns:
            Two numpy arrays: the first one, representing the binary mask, and the second one, representing the original image with the binary mask drawn over it.
        """
        if image.ndim == 2:
            resized_image_normalized = skimage.transform.resize(image, (self.__input_size[0], self.__input_size[1]))
        else:
            resized_image_normalized = skimage.transform.resize(image, (self.__input_size[0], self.__input_size[1], image.shape[2]))

        resized_image = cv2.resize(image, (self.__input_size[0], self.__input_size[1]))

        prediction = self.__internal_model.model.predict(numpy.array([resized_image_normalized]))
        prediction_mask = prediction[0] >= binary_threshold
        prediction_mask_int = 255 * prediction_mask

        for x in range(0, prediction_mask.shape[0]):
            for y in range(0, prediction_mask.shape[1]):
                if prediction_mask[x][y]:
                    resized_image[x, y, 0] = 0
                    resized_image[x, y, 1] = 255
                    resized_image[x, y, 2] = 0

        resized_image = cv2.resize(resized_image, (image.shape[1], image.shape[0]))

        return prediction_mask_int, resized_image

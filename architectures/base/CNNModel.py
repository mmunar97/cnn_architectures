from tensorflow import keras
from typing import Tuple


class CNNModel:

    def __init__(self,
                 input_size: Tuple[int, ...]):
        self.__input_size: Tuple[int, ...] = None
        self.__internal_model: keras.models.Model = None

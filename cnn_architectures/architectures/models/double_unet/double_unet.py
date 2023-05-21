from tensorflow.keras.applications import *
from typing import Tuple

from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.double_unet.layers import *

import tensorflow.keras.models as keras_model


class DoubleUNet(CNNModel):

    def __init__(self, input_size: Tuple[int, int, int]):
        super().__init__(input_size)

        self.__history = None

    def build(self):
        input_image = keras_layer.Input(self.input_size, name="input_image")

        vgg = VGG19(include_top=False, weights='imagenet', input_tensor=input_image)

        x, skip1 = VGGEncoder(vgg_model=vgg)(input_image)
        x = AtrousSpatialPyramidPooling(n_filters=64)(x)
        x = ForwardConnectedDecoder(connections=skip1)(x)

        output1 = OutputBlock()(x)
        x = input_image * output1

        x, skip2 = ForwardEncoder()(x)
        x = AtrousSpatialPyramidPooling(n_filters=64)(x)
        x = ForwardDoubleConnectedDecoder(connections1=skip1, connections2=skip2)(x)

        output2 = OutputBlock()(x)
        final_output = Concatenate(name="final_output")([output1, output2])

        model = keras_model.Model(inputs=input_image, outputs=final_output)
        self.set_model(model)

        return input_image, final_output
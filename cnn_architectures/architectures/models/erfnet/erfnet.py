from tensorflow import keras
from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.erfnet.layers import *
from cnn_architectures.utils.common import DownSamplerBlock
from typing import Union, Tuple
from tensorflow.keras.layers import Input, Conv2DTranspose
from cnn_architectures.utils.common import ConvBlock as UpSamplerBlock


class ERFNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 out_channels: int,
                 last_activation: str = 'softmax'):
        super(ERFNet, self).__init__(input_size=input_size)
        self.__last_activation = last_activation
        self.__out_channels = out_channels
        self.__history = None

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        x = input_image
        x = DownSamplerBlock(nIn=self.input_size[-1], nOut=16)(x)
        x = DownSamplerBlock(nIn=16, nOut=64)(x)

        for _ in range(5):
            x = NBN1D(filters=64)(x)

        x = DownSamplerBlock(nIn=64, nOut=128)(x)
        for d in [2, 4, 8, 16, 2, 4, 8, 16]:
            x = NBN1D(filters=128, dilation=d)(x)

        for f in [64, 16]:
            x = UpSamplerBlock(filters=f, kSize=3, strides=2, conv_type='trans')(x)
            for _ in range(2):
                x = NBN1D(filters=f)(x)

        mask_out = UpSamplerBlock(filters=self.__out_channels,
                                  kSize=2,
                                  strides=2,
                                  conv_type='trans',
                                  activation_function=self.__last_activation,
                                  name='mask_out')(x)

        model = keras.models.Model(inputs=input_image, outputs=mask_out)
        self.set_model(model)
        return mask_out
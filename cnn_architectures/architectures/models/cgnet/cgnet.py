from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.cgnet.layers import *
from typing import Union, Tuple
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.models import Model


class CGNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 m: int = 3,
                 n: int = 21,
                 out_channels: int = 2,
                 last_activation: str = 'softmax'):
        super(CGNet, self).__init__(input_size=input_size)
        self.__last_activation = last_activation
        self.__out_channels = out_channels
        self.__m = m
        self.__n = n
        self.__history = None

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        inp = input_image

        # Stage 1
        x = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   strides=2,
                   padding='same')(inp)
        x_s1 = x
        for _ in range(2):
            x = Conv2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same')(x)
        ij_1 = InputInjection(downsamplingratio=1)(inp)

        x = concatenate([x_s1, ij_1, x])

        x = CGBlock(filters=64,
                    dilation=2,
                    r=8, n_block=1)(x)
        x_s2 = x
        for i in range(self.__m - 1):
            x = CGBlock(filters=64,
                        dilation=2,
                        r=8,
                        n_block=i+2)(x)

        ij_2 = InputInjection(downsamplingratio=2)(inp)

        x = concatenate([x_s2, ij_2, x])

        x = CGBlock(filters=128,
                    dilation=4,
                    r=16,
                    n_block=1)(x)
        x_s3 = x
        for i in range(self.__n-1):
            x = CGBlock(filters=128,
                        dilation=4,
                        r=16,
                        n_block=i+2)(x)

        x = concatenate([x, x_s3])
        x = Conv2D(filters=self.__out_channels,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same')(x)

        x = UpSampling2D(size=(8, 8),
                         interpolation='bilinear')(x)
        mask_out = Activation(self.__last_activation, name='mask_out')(x)

        model = Model(inputs=input_image, outputs=mask_out, name='CGNet')
        self.set_model(model)
        return mask_out
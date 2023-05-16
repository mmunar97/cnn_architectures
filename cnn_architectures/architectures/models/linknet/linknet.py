from tensorflow.keras.models import Model
from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.linknet.layers import *
from cnn_architectures.utils.common import ConvBlock
from typing import Union, Tuple
from tensorflow.keras.layers import Input, MaxPooling2D, add


class LinkNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 out_channels: int,
                 last_activation: str = 'softmax'):

        super(LinkNet, self).__init__(input_size=input_size)
        self.__last_activation = last_activation
        self.__out_channels = out_channels
        self.__history = None

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        x = input_image

        x = ConvBlock(filters=64,
                      kSize=7,
                      strides=2)(x)
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=2,
                         padding='same')(x)
        x1 = Encoder(m=64, n=64)(x)
        x2 = Encoder(m=64, n=128)(x1)
        x3 = Encoder(m=128, n=256)(x2)
        x4 = Encoder(m=256, n=512)(x3)

        x = Decoder(m=512, n=256)(x4)
        x = add([x, x3])
        x = Decoder(m=256, n=128)(x)
        x = add([x, x2])
        x = Decoder(m=128, n=64)(x)
        x = add([x, x1])
        x = Decoder(m=64, n=64)(x)

        x = ConvBlock(filters=32,
                      kSize=3,
                      strides=2,
                      conv_type='trans')(x)
        x = ConvBlock(filters=32,
                      kSize=3)(x)
        mask_out = ConvBlock(filters=self.__out_channels,
                             kSize=2,
                             strides=2,
                             conv_type='trans',
                             activation_function=self.__last_activation,
                             name='mask_out')(x)

        model = Model(inputs=input_image, outputs=mask_out, name='LinkNet')

        self.set_model(model)

        return mask_out
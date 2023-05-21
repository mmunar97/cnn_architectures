from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB4

from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.unet_pre.layers import *
from typing import Union, Tuple
from tensorflow.keras.layers import Input

class UNetPre(CNNModel):
    def __init__(self,
                 input_size,
                 out_channels: int):
        super(UNetPre, self).__init__(input_size=input_size)

        self.__out_channels = out_channels
        self.__history = None
        backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_size)
        self.__backbone = [
            backbone.layers[31],
            backbone.layers[90],
            backbone.layers[149],
            backbone.layers[326]
        ]
        self.__filters = [3, 144, 192, 336, 960, 512, 256, 128, 64, 32, 16, 1]

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        x = input_image

        encoders = []

        for idx, bb in enumerate(self.__backbone):
            x = DownSampling(self.__filters[idx+1])(x)
            x = EncoderBlock(bb)(x)
            encoders.append(x)

        x = SpecialDownSampling()(x)
        x = SpecialEncoderBlock(self.__filters[5])(x)

        for filter in self.__filters[6:10]:
            x = UpSampling(filter)(x)
            x = DecoderBlock(filter)(x, conc=encoders.pop(-1))

        x = UpSampling(self.__filters[10])(x)

        x = Dropout(0.1)(x)
        x = Conv2D(filters=self.__filters[10], kernel_size=(3, 3), padding='same')(x)
        x = ResidualBlock(self.__filters[10])(x)
        x = ResidualBlock(self.__filters[10])(x)
        x = LeakyReLU()(x)

        mask_out = Conv2D(filters=1,
                          kernel_size=(1, 1),
                          activation='sigmoid',
                          padding='same',
                          name='mask_out')(x)

        model = Model(inputs=input_image, outputs=mask_out, name='UnetPre')
        self.set_model(model)
        return mask_out

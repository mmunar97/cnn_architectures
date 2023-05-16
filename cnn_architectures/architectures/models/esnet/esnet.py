from tensorflow import keras
from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.esnet.layers import *
from cnn_architectures.utils.common import DownSamplerBlock, UpSamplerBlock
from typing import Union, Tuple
from tensorflow.keras.layers import Input, Conv2DTranspose


class ESNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 out_channels: int,
                 last_activation: str = 'softmax'):
        super(ESNet, self).__init__(input_size=input_size)
        self.__last_activation = last_activation
        self.__out_channels = out_channels
        self.__history = None

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        x = input_image

        x = DownSamplerBlock(nIn=self.input_size[-1], nOut=16)(x)
        for _ in range(3):
            x = FCU(k=3, filters=16, dropout_rate=0.3)(x)

        x = DownSamplerBlock(nIn=16, nOut=64)(x)
        for _ in range(2):
            x = FCU(k=5, filters=64, dropout_rate=0.3)(x)

        x = DownSamplerBlock(nIn=64, nOut=128)(x)
        for _ in range(3):
            x = PFCU(filters=128, dropout_rate=0.3)(x)

        x = UpSamplerBlock(filters=64)(x)
        for _ in range(2):
            x = FCU(k=5, filters=64, dropout_rate=0.3)(x)

        x = UpSamplerBlock(filters=16)(x)
        for _ in range(2):
            x = FCU(k=3, filters=16, dropout_rate=0.3)(x)

        mask_out = Conv2DTranspose(filters=self.__out_channels,
                                   kernel_size=(2, 2),
                                   strides = 2,
                                   padding='same',
                                   use_bias=True,
                                   activation=self.__last_activation,
                                   name='mask_out')(x)

        # x = UpSamplerBlock(filters=self.__out_channels)(x)
        # mask_out = Activation(self.__last_activation, name='mask_out')(x)

        model = keras.models.Model(inputs=input_image, outputs=mask_out, name='ESNet')

        self.set_model(model)
        return mask_out


if __name__ == '__main__':
    model = ESNet(input_size=(256, 256, 3), out_channels=2)
    model.build()
    model.compile()
    model.summary()

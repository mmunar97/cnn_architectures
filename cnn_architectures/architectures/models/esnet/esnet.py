from tensorflow.keras.models import Model
from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.esnet.layers import *
from cnn_architectures.utils.common import DownSamplerBlock
from cnn_architectures.utils.common import ConvBlock as UpSamplerBlock
from typing import Union, Tuple
from tensorflow.keras.layers import Input


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

        x = UpSamplerBlock(filters=64, kSize=3, strides=2, conv_type='trans')(x)
        for _ in range(2):
            x = FCU(k=5, filters=64, dropout_rate=0.3)(x)

        x = UpSamplerBlock(filters=16, kSize=3, strides=2, conv_type='trans')(x)
        for _ in range(2):
            x = FCU(k=3, filters=16, dropout_rate=0.3)(x)

        mask_out = UpSamplerBlock(filters=self.__out_channels,
                                  kSize=2,
                                  strides=2,
                                  conv_type='trans',
                                  activation_function=self.__last_activation,
                                  name='mask_out')(x)

        model = Model(inputs=input_image, outputs=mask_out, name='ESNet')

        self.set_model(model)
        return mask_out
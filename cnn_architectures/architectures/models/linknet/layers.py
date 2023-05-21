from tensorflow.keras.layers import Layer, add, AvgPool2D
from cnn_architectures.utils.common import ConvBlock, Identity





class EncoderSubBlock(Layer):
    def __init__(self, nIn: int, nOut: int, block_number: int):
        super(EncoderSubBlock, self).__init__()
        self.__in_filters = nIn
        self.__out_filters = nOut
        self.__block_number = block_number

        if block_number == 1:
            self.block1 = ConvBlock(filters=nOut, kSize=3, strides=2)
            if nIn == nOut:
                self.residual = AvgPool2D(pool_size=(2, 2), padding='same')
            else:
                self.residual = ConvBlock(filters=nOut, kSize=1, strides=2)
        else:
            self.block1 = ConvBlock(filters=nOut, kSize=3)
            self.residual = Identity()
        self.block2 = ConvBlock(filters=nOut, kSize=3)

    def call(self, inputs, *args, **kwargs):
        out = self.block1(inputs)
        out = self.block2(out)
        residual_input = self.residual(inputs)
        out = add([out, residual_input])

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_filters': self.__in_filters,
            'out_filters': self.__out_filters,
            'block_numer': self.__block_number,
        })
        return config


class Encoder(Layer):
    def __init__(self, m: int, n: int):
        super(Encoder, self).__init__()
        self.__m = m
        self.__n = n

        self.block1 = EncoderSubBlock(nIn=m, nOut=n, block_number=1)
        self.block2 = EncoderSubBlock(nIn=n, nOut=n, block_number=2)

    def call(self, inputs, *args, **kwargs):
        out1 = self.block1(inputs)
        out2 = self.block2(out1)
        return out2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'm': self.__m,
            'n': self.__n,
        })
        return config


class Decoder(Layer):
    def __init__(self, m: int, n: int):
        super(Decoder, self).__init__()
        self.__in_filters = m
        self.__out_filters = n

        self.block1 = ConvBlock(filters=(int(m/4)),
                                kSize=1)
        self.block2 = ConvBlock(filters=(int(m/4)),
                                kSize=3,
                                strides=2,
                                conv_type='trans')
        self.block3 = ConvBlock(filters=n,
                                kSize=1)
    
    def call(self, inputs, *args, **kwargs):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        return out
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_filters': self.__in_filters,
            'out_filters': self.__out_filters,
        })
        return config

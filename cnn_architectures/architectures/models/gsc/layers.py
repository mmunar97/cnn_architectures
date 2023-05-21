from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, multiply, add
from cnn_architectures.utils.common import ConvBlock


class EncoderBlock(Layer):

    def __init__(self, number_filters: int):
        super(EncoderBlock, self).__init__()
        self.__number_filters = number_filters

        self.conv_block1 = ConvBlock(filters=number_filters, kSize=3)
        self.conv_block2 = ConvBlock(filters=number_filters, kSize=3)
        self.mp = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.conv_block1(inputs)
        y = self.conv_block2(x)
        z = self.mp(y)
        return z, y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'number_filters': self.__number_filters
        })
        return config


class GatedConvNet(Layer):

    def __init__(self):
        super(GatedConvNet, self).__init__()
        self.conv = Conv2D(filters=1,
                           kernel_size=(1, 1),
                           activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        x = concatenate([inputs[0], inputs[1]])
        x = self.conv(x)
        return x


class DecoderBlock(Layer):

    def __init__(self, number_filters: int):
        super(DecoderBlock, self).__init__()

        self.__number_filters = number_filters
        self.conv_trans = Conv2DTranspose(number_filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')
        self.gcn = GatedConvNet()
        self.conv_block1 = ConvBlock(filters=number_filters, kSize=3)
        self.conv_block2 = ConvBlock(filters=number_filters, kSize=3)

    def call(self, inputs, *args, **kwargs):
        x = self.conv_trans(inputs[0])
        y = self.gcn([inputs[1], x])
        z = multiply([inputs[1], y])
        x = add([x, z])
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'number_filters': self.__number_filters
        })
        return config
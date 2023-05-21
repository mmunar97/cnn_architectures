import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MaxPooling2D, Conv2D, LeakyReLU, Dropout, BatchNormalization, Conv2DTranspose, concatenate, add

class EncoderBlock(Layer):
    def __init__(self, bb):
        super(EncoderBlock, self).__init__()
        self.__bb = bb

    def call(self, inputs, *args, **kwargs):
        x = self.bb(inputs)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'backbone': self.__bb,
        })
        return config

class DownSampling(Layer):
    def __init__(self, filter):
        super(DownSampling, self).__init__()
        self.__filters = filter

        self.mp = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv = Conv2D(filters=filter, kernel_size=(3, 3), padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.mp(inputs)
        x = self.conv(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config

class SpecialDownSampling(Layer):
    def __init__(self):
        super(SpecialDownSampling, self).__init__()
        self.lr = LeakyReLU()
        self.mp = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.do = Dropout(0.1)

    def call(self, inputs, *args, **kwargs):
        x = self.lr(inputs)
        x = self.mp(x)
        x = self.do(x)
        return x

class ResidualBlock(Layer):
    def __init__(self, filter):
        super(ResidualBlock, self).__init__()
        self.__filters = filter

        self.lr1 = LeakyReLU()
        self.lr2 = LeakyReLU()

        self.bn = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

        self.conv1 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), padding='same')
        self.conv2 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), padding='same')


    def call(self, inputs, *args, **kwargs):
        x = self.lr1(inputs)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.lr2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = add([x, self.bn(inputs)])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config

class SpecialEncoderBlock(Layer):
    def __init__(self, filter):
        super(SpecialEncoderBlock, self).__init__()
        self.__filters = filter

        self.conv = Conv2D(filters=filter, kernel_size=(3, 3), padding='same')
        self.res1 = ResidualBlock(filter)
        self.res2 = ResidualBlock(filter)
        self.lr = LeakyReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.lr(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config

class UpSampling(Layer):
    def __init__(self, filter):
        super(UpSampling, self).__init__()
        self.__filters = filter

        self.convt = Conv2DTranspose(filter, kernel_size=(2, 2), strides=(2, 2), padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.convt(inputs)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config

class DecoderBlock(Layer):
    def __init__(self, filter):
        super(DecoderBlock, self).__init__()
        self.__filters = filter

        self.do = Dropout(0.1)
        self.conv = Conv2D(filters=filter, kernel_size=(3, 3), padding='same')
        self.rb1 = ResidualBlock(filter)
        self.rb2 = ResidualBlock(filter)
        self.lr = LeakyReLU()

    def call(self, inputs, *args, **kwargs):
        x = concatenate([kwargs.get('conc'), inputs])
        x = self.do(x)
        x = self.conv(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.lr(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config
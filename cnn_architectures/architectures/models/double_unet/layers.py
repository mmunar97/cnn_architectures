from tensorflow.keras.layers import Layer, AveragePooling2D, UpSampling2D, Conv2D, BatchNormalization, concatenate, multiply, Dense, Reshape, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.activations import relu
from cnn_architectures.utils.common import ConvBlock



class ASPP(Layer):
    def __init__(self, filters, input_shape):
        super(ASPP, self).__init__()
        self.__filters = filters
        self.__input_shape = input_shape

        self.avgpool = AveragePooling2D(pool_size=(input_shape[1], input_shape[2]))
        self.up = UpSampling2D(size=(input_shape[1], input_shape[2]), interpolation='bilinear')
        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            dilation_rate=1,
                            padding='same',
                            use_bias=False)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=6,
                            padding='same',
                            use_bias=False)
        self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=12,
                            padding='same',
                            use_bias=False)
        self.bn4 = BatchNormalization()

        self.conv5 = Conv2D(filters=filters,
                            kernel_size=(3, 3),
                            dilation_rate=18,
                            padding='same',
                            use_bias=False)
        self.bn5 = BatchNormalization()

        self.conv = Conv2D(filters=filters,
                           kernel_size=(1, 1),
                           padding='same',
                           use_bias=False)
        self.bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        y1 = self.avgpool(inputs)
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = relu(y1)
        y1 = self.up(y1)

        y2 = self.conv2(inputs)
        y2 = self.bn2(y2)
        y2 = relu(y2)

        y3 = self.conv3(inputs)
        y3 = self.bn3(y3)
        y3 = relu(y3)

        y4 = self.conv4(inputs)
        y4 = self.bn4(y4)
        y4 = relu(y4)

        y5 = self.conv5(inputs)
        y5 = self.bn5(y5)
        y5 = relu(y5)
        y = concatenate([y1, y2, y3, y4, y5])

        y = self.conv(y)
        y = self.bn(y)
        y = relu(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'input_shape': self.__input_shape,
        })
        return config


class SubEncoder(Layer):
    def __init__(self, filters):
        super(SubEncoder, self).__init__()
        self.__filters = filters

        self.conv_block1 = ConvBlock(filters=filters,
                                     kSize=3)
        self.conv_block2 = ConvBlock(filters=filters,
                                     kSize=3)
        self.squeeze = Squeeze_excite_block(filters=filters)

    def call(self, inputs, **kwargs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.squeeze(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters
        })
        return config


class Squeeze_excite_block(Layer):
    def __init__(self, filters: int, ratio: int = 8):
        super(Squeeze_excite_block, self).__init__()
        self.__filters = filters
        self.__ratio = ratio

        self.gap = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, filters))
        self.dense1 = Dense(units=filters//ratio,
                            activation='relu',
                            kernel_initializer='he_normal',
                            use_bias=False)
        self.dense2 = Dense(units=filters,
                            activation='sigmoid',
                            kernel_initializer='he_normal',
                            use_bias=False)

    def call(self, inputs, **kwargs):
        init = inputs
        x = self.gap(init)
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = multiply([init, x])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'ratio': self.__ratio,
        })
        return config

class OutputBlock(Layer):
    def __init__(self):
        super(OutputBlock, self).__init__()
        self.conv = Conv2D(filters=1,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        return x

class Encoder(Layer):
    def __init__(self, filters: int):
        super(Encoder, self).__init__()
        self.__filters = filters
        self.conv_block = SubEncoder(filters=filters)
        self.mp = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs, **kwargs):
        x = self.conv_block(inputs)
        out = self.mp(x)
        return out, x
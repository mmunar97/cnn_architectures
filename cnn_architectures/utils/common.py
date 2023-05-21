from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, BatchNormalization, concatenate, ReLU, Conv2DTranspose, Activation

# LinkNet / DoubleUNet
class Identity(Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inp = inputs
        return inp


# ESNet/ERFNet
class DownSamplerBlock(Layer):
    def __init__(self, nIn: int, nOut: int):
        """

        Args:
            nIn:
            nOut:
        """
        super(DownSamplerBlock, self).__init__()
        self.__in_filters = nIn
        self.__out_filters = nOut
        self.conv = Conv2D(filters=nOut-nIn,
                           kernel_size=(3, 3),
                           strides=2,
                           padding='same')
        self.pool = MaxPooling2D(pool_size=(2, 2),
                                 strides=2)

        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs:
            **kwargs:

        Returns:

        """
        inp = inputs
        out1 = self.conv(inp)
        out2 = self.pool(inp)
        out = concatenate([out1, out2])
        out = self.bn(out)
        out = self.relu(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_filters': self.__in_filters,
            'out_filters': self.__out_filters,
        })
        return config


# ESNet/ERFNet
class UpSamplerBlock(Layer):
    def __init__(self, filters):
        super(UpSamplerBlock, self).__init__()
        self.__filters = filters
        self.tconv = Conv2DTranspose(filters=filters,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        out = self.tconv(inputs)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
        })
        return config


# LinkNet / GSC / ESNet / ERFNet
class ConvBlock(Layer):
    def __init__(self,
                 filters: int,
                 kSize: int,
                 conv_type: str = 'normal',
                 strides: int = 1,
                 activation_function: str = 'relu', **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.__filters = filters
        self.__strides = strides
        self.__kernel_size = kSize
        self.__conv_type = conv_type
        self.__activation_function = activation_function

        if conv_type == 'trans':
            self.conv = Conv2DTranspose(filters=filters,
                                        kernel_size=(kSize, kSize),
                                        strides=strides,
                                        padding='same',
                                        )
        else:
            self.conv = Conv2D(filters=filters,
                               kernel_size=(kSize, kSize),
                               strides=strides,
                               padding='same',
                               )
        self.bn = BatchNormalization()
        self.activation = Activation(self.__activation_function)

    def call(self, inputs, *args, **kwargs):
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.activation(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'strides': self.__strides,
            'kernel_size': self.__kernel_size,
            'conv_type': self.__conv_type,
            'activation_type': self.__activation_function,
        })
        return config

from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, PReLU, GlobalAvgPool2D, AveragePooling2D, Dense, multiply, concatenate


class FLoc(Layer):
    def __init__(self, dm: int):
        super(FLoc, self).__init__()
        self.__depth_multiplier = dm
        self.conv = DepthwiseConv2D(kernel_size=(3, 3),
                                    padding='same',
                                    depth_multiplier=dm,
                                    )

    def call(self, inputs, **kwargs):
        out = self.conv(inputs)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth_multiplier': self.__depth_multiplier,
        })
        return config


class FSur(Layer):
    def __init__(self, dm: int, dilation: int):
        super(FSur, self).__init__()
        self.__depth_multiplier = dm
        self.__dilation = dilation
        self.conv = DepthwiseConv2D(kernel_size=(3, 3),
                                    padding='same',
                                    depth_multiplier=dm,
                                    dilation_rate=dilation
                                    )

    def call(self, inputs, **kwargs):
        out = self.conv(inputs)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth_multiplier': self.__depth_multiplier,
            'dilation_rate': self.__dilation,
        })
        return config


class FGlo(Layer):
    def __init__(self, filters, reduction):
        super(FGlo, self).__init__()
        self.__filters = filters
        self.__reduction = reduction
        self.pool = GlobalAvgPool2D()
        self.mlp = MultiLayerPerceptron(filters=filters, reduction=reduction)

    def call(self, inputs, **kwargs):
        inp = inputs
        out = self.pool(inp)
        out = self.mlp(out)
        out = multiply([inp, out])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'reduction': self.__reduction,
        })
        return config


class MultiLayerPerceptron(Layer):
    def __init__(self, filters: int, reduction: int):
        super(MultiLayerPerceptron, self).__init__()
        self.__filters = filters
        self.__reduction = reduction
        self.dense1 = Dense(units=filters // reduction, activation='relu')
        self.dense2 = Dense(units=filters, activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        out = self.dense1(inputs)
        out = self.dense2(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'reduction': self.__reduction,
        })
        return config


class CGBlock(Layer):
    def __init__(self, filters: int, r: int, dilation: int, n_block: int):
        super(CGBlock, self).__init__()
        self.__filters = filters
        self.__dilation = dilation
        self.__n_block = n_block
        if n_block == 1:
            self.conv = Conv2D(filters=filters/2,
                               kernel_size=(1, 1),
                               padding='same',
                               strides=2)
        else:
            self.conv = Conv2D(filters=filters / 2,
                               kernel_size=(1, 1),
                               padding='same')
        self.floc = FLoc(dm=1)
        self.fsur = FSur(dm=1, dilation=dilation)
        self.fglo = FGlo(filters=filters,
                         reduction=r)

        self.bn = BatchNormalization()
        self.prelu = PReLU()

    def call(self, inputs, **kwargs):
        mid = self.conv(inputs)
        out1 = self.floc(mid)
        out2 = self.fsur(mid)
        out = concatenate([out1, out2])
        out = self.bn(out)
        out = self.prelu(out)
        out = self.fglo(out)
        # Global residual learning
        out = concatenate([mid, out])

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'dilated_conv_rate': self.__dilation,
            'n_block': self.__n_block,
        })
        return config


class InputInjection(Layer):
    def __init__(self, downsamplingratio):
        super(InputInjection, self).__init__()
        self.pool = []
        for _ in range(0, downsamplingratio):
            self.pool.append(AveragePooling2D(pool_size=(3, 3),
                                              strides=2,
                                              padding='same'))

    def call(self, inputs, *args, **kwargs):
        inp = inputs
        for pool in self.pool:
            inp = pool(inp)
        return inp
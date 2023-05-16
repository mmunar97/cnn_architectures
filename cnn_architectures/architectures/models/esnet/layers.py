from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, add, Dropout


class FCU(Layer):
    def __init__(self, k: int, filters: int, dropout_rate: float = 0):
        super(FCU, self).__init__()
        self.__dropout_rate = dropout_rate
        self.__k = k
        self.__filters = filters

        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(k, 1),
                            padding='same',
                            activation='relu')
        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, k),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv3 = Conv2D(filters=filters,
                            kernel_size=(k, 1),
                            padding='same',
                            activation='relu')
        self.conv4 = Conv2D(filters=filters,
                            kernel_size=(1, k),
                            padding='same')
        self.bn2 = BatchNormalization()

        self.relu2 = ReLU()

        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        inp = inputs

        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn2(out)

        if self.dropout.rate != 0:
            out = self.dropout(out)

        out = add([out, inp])
        out = self.relu2(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv_kernel': self.__k,
            'filters': self.__filters,
            'dropout_rate': self.__dropout_rate
        })
        return config


class DilatedConvBlock(Layer):
    def __init__(self, filters: int, dilation: int, dropout_rate: float = 0):
        super(DilatedConvBlock, self).__init__()
        self.__dropout_rate = dropout_rate
        self.__filters = filters
        self.__dilation = dilation

        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(3, 1),
                            padding='same',
                            dilation_rate=(dilation, 1),
                            activation='relu')
        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, 3),
                            padding='same',
                            dilation_rate=(1, dilation))
        self.bn = BatchNormalization()

        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.bn(out)
        if self.dropout.rate != 0:
            out = self.dropout(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'dilation': self.__dilation,
            'dropout_rate': self.__dropout_rate
        })
        return config


class PFCU(Layer):
    def __init__(self, filters: int, dropout_rate: float = 0):
        super(PFCU, self).__init__()
        self.__filters = filters
        self.__dropout_rate = dropout_rate

        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(3, 1),
                            padding='same',
                            activation='relu')
        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, 3),
                            padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

        self.dblock1 = DilatedConvBlock(filters=filters, dilation=2, dropout_rate=dropout_rate)
        self.dblock2 = DilatedConvBlock(filters=filters, dilation=5, dropout_rate=dropout_rate)
        self.dblock3 = DilatedConvBlock(filters=filters, dilation=9, dropout_rate=dropout_rate)

        self.relu2 = ReLU()

    def call(self, inputs, **kwargs):
        inp = inputs
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out1 = self.dblock1(out)
        out2 = self.dblock2(out)
        out3 = self.dblock3(out)

        out = add([inp, out1, out2, out3])
        out = self.relu2(out)

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.__filters,
            'dropout_rate': self.__dropout_rate,
        })
        return config
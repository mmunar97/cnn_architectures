from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, add, Dropout

class NBN1D(Layer):
    def __init__(self, filters: int, dilation: int=1):
        super(NBN1D, self).__init__()
        self.__filters = filters
        self.__dilation = dilation

        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(3, 1),
                            padding='same',
                            activation='relu')

        self.conv2 = Conv2D(filters=filters,
                            kernel_size=(1, 3),
                            padding='same',
                            activation='relu')

        self.bn1 = BatchNormalization()

        self.conv3 = Conv2D(filters=filters,
                            kernel_size=(3, 1),
                            padding='same',
                            dilation_rate=(dilation, 1),
                            activation='relu')

        self.conv4 = Conv2D(filters=filters,
                            kernel_size=(1, 3),
                            padding='same',
                            dilation_rate=(1, dilation))

        self.bn2 = BatchNormalization()

        self.drop = Dropout(0.3)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        inp = inputs
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn2(out)
        out = self.drop(out)
        out = add([inp, out])
        out = self.relu(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dilation': self.__dilation,
            'filters': self.__filters,
        })
        return config
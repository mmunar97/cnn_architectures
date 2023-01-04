from tensorflow import keras


class ConvBlock(keras.layers.Layer):

    def __init__(self, number_filters: int):
        super(ConvBlock, self).__init__()
        self.conv = keras.layers.Conv2D(filters=number_filters, kernel_size=(3, 3), padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(keras.layers.Layer):

    def __init__(self, number_filters: int):
        super(EncoderBlock, self).__init__()
        self.conv_block1 = ConvBlock(number_filters)
        self.conv_block2 = ConvBlock(number_filters)
        self.mp = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.conv_block1(inputs)
        y = self.conv_block2(x)
        z = self.mp(y)
        return z, y


class GatedConvNet(keras.layers.Layer):

    def __init__(self):
        super(GatedConvNet, self).__init__()
        self.conc = keras.layers.Concatenate()
        self.conv = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        x = self.conc([inputs[0], inputs[1]])
        x = self.conv(x)
        return x


class DecoderBlock(keras.layers.Layer):

    def __init__(self, number_filters):
        super(DecoderBlock, self).__init__()
        self.convtrans = keras.layers.Conv2DTranspose(number_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.gcn = GatedConvNet()
        self.mult = keras.layers.Multiply()
        self.add = keras.layers.Add()
        self.convblock1 = ConvBlock(number_filters)
        self.convblock2 = ConvBlock(number_filters)

    def call(self, inputs, *args, **kwargs):
        x = self.convtrans(inputs[0])
        y = self.gcn([inputs[1], x])
        z = self.mult([inputs[1], y])
        x = self.add([x, z])
        x = self.convblock1(x)
        x = self.convblock2(x)
        return x

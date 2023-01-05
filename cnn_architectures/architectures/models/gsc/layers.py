from tensorflow.keras import layers


class ConvBlock(layers.Layer):

    def __init__(self, number_filters: int):
        super(ConvBlock, self).__init__()
        self.number_filters = number_filters

        self.conv = layers.Conv2D(filters=self.number_filters, kernel_size=(3, 3), padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        self.add_weight("ConvBlock")
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'number_filters': self.number_filters
        })


class EncoderBlock(layers.Layer):

    def __init__(self, number_filters: int):
        super(EncoderBlock, self).__init__()
        self.number_filters = number_filters

        self.conv_block1 = ConvBlock(self.number_filters)
        self.conv_block2 = ConvBlock(self.number_filters)
        self.mp = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

    def call(self, inputs, *args, **kwargs):
        self.add_weight("EncoderBlock")
        x = self.conv_block1(inputs)
        y = self.conv_block2(x)
        z = self.mp(y)
        return z, y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'number_filters': self.number_filters
        })


class GatedConvNet(layers.Layer):

    def __init__(self):
        super(GatedConvNet, self).__init__()
        self.conc = layers.Concatenate()
        self.conv = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        self.add_weight("GatedConvNet")
        x = self.conc([inputs[0], inputs[1]])
        x = self.conv(x)
        return x


class DecoderBlock(layers.Layer):

    def __init__(self, number_filters):
        super(DecoderBlock, self).__init__()

        self.number_filters = number_filters
        self.convtrans = layers.Conv2DTranspose(self.number_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.gcn = GatedConvNet()
        self.mult = layers.Multiply()
        self.add = layers.Add()
        self.convblock1 = ConvBlock(self.number_filters)
        self.convblock2 = ConvBlock(self.number_filters)

    def call(self, inputs, *args, **kwargs):
        self.add_weight("DecoderBlock")
        x = self.convtrans(inputs[0])
        y = self.gcn([inputs[1], x])
        z = self.mult([inputs[1], y])
        x = self.add([x, z])
        x = self.convblock1(x)
        x = self.convblock2(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'number_filters': self.number_filters
        })
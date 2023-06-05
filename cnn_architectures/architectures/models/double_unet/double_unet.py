from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.double_unet.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, concatenate
from tensorflow.keras.applications import VGG19
from typing import Tuple


class DoubleUNet(CNNModel):
    def __init__(self, input_size, filters: Tuple[int, int, int, int] = (256, 128, 64, 32)):
        super(DoubleUNet, self).__init__(input_size=input_size)
        self.__history = None
        self.__filters = filters
        self.__vgg19 = None

        self.up = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def build(self):
        input_image = Input(self.input_size, name='input_image')
        self.__vgg19 = VGG19(include_top=False,
                             weights='imagenet',
                             input_tensor=input_image)

        # Encoder 1
        x = self.__vgg19.get_layer("block5_conv4").output

        x = ASPP(filters=64, input_shape=x.shape)(x)

        # Decoder Blocks
        layer_names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for idx, filters in enumerate(self.__filters):
            x = self.up(x)
            x = concatenate([x, self.__vgg19.get_layer(layer_names[4 - (idx+1)]).output])
            x = SubEncoder(filters=filters)(x)

        output1 = OutputBlock()(x)

        x = multiply([input_image, output1])

        skip_connections_2 = []
        for filters in reversed(self.__filters):
            x, skip2 = Encoder(filters=filters)(x)
            skip_connections_2.append(skip2)

        x = ASPP(filters=64, input_shape=x.shape)(x)

        for idx, filters in enumerate(self.__filters):
            x = self.up(x)
            x = concatenate([x, self.__vgg19.get_layer(layer_names[4 - (idx+1)]).output, skip_connections_2[4 - (idx + 1)]])
            x = SubEncoder(filters=filters)(x)

        output2 = OutputBlock()(x)
        mask_out = concatenate([output1, output2], name='mask_out')

        model = Model(inputs=input_image, outputs=mask_out)
        self.set_model(model)

        return input_image, mask_out

if __name__=='__main__':
    model = DoubleUNet(input_size=(256, 256, 3))
    model.build()
    model.summary()
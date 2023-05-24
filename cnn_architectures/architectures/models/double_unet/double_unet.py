from cnn_architectures.architectures.models.double_unet.layers import *
from cnn_architectures.architectures.base.CNNModel import CNNModel
from typing import Union, Tuple
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from cnn_architectures.utils.common import ConvBlock

class DoubleUNet(CNNModel):
    def __init__(self,
                 input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 # out_channels: int,
                 last_activation: str = 'softmax'):

        super(DoubleUNet, self).__init__(input_size=input_size)
        self.__last_activation = last_activation
        # self.__out_channels = out_channels
        self.__history = None
        self.vgg19 = None
        self.filters = [256, 128, 64, 32]
        self.conv_blocks1 = []
        for f in self.filters:
            self.conv_blocks1.append(SubEncoder(filters=f))
        self.up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.mp = MaxPooling2D(pool_size=(2, 2))

        self.conv_blocks2 = []
        for f in reversed(self.filters):
            self.conv_blocks2.append(SubEncoder(filters=f))

        self.conv_blocks3 = []
        for f in self.filters:
            self.conv_blocks3.append(SubEncoder(filters=f))


    def build(self):
        input_image = Input(self.input_size, name='input_image')
        self.vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=input_image)
        # Encoder 1
        x = self.vgg19.get_layer('block5_conv4').output

        x = ASPP(64, input_shape=x.shape)(x)

        # Decoder 1
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        skip1 = []
        for name in names:
            skip1.append(self.vgg19.get_layer(name).output)

        for i in range(len(self.conv_blocks1)):
            x = self.up(x)
            x = concatenate([x, skip1[len(skip1) - (i+1)]])
            x = self.conv_blocks1[i](x)

        outputs1 = OutputBlock()(x)

        x = multiply([input_image, outputs1])

        # Encoder 2
        skip2 = []
        for element in self.conv_blocks2:
            x = element(x)
            skip2.append(x)
            x = self.mp(x)

        x = ASPP(64, input_shape=x.shape)(x)

        # Decoder 2
        for i in range(len(skip1)):
            x = self.up(x)
            x = concatenate([x, skip1[len(skip1) - (i+1)], skip2[len(skip2) - (i+1)]])
            x = self.conv_blocks3[i](x)

        outputs2 = OutputBlock()(x)
        mask_out = concatenate([outputs1, outputs2], name='mask_out')

        model = Model(inputs=input_image, outputs=mask_out, name='DoubleUNet')

        self.set_model(model)

        return mask_out

from cnn_architectures.utils.metrics import dice_coef, dice_loss

if __name__ == '__main__':
    model = DoubleUNet(input_size=(256, 256, 3))
    model.build()
    model.compile(learning_rate=0.0001,
                  loss_func=dice_loss,
                  metrics=[dice_coef])
    model.summary()
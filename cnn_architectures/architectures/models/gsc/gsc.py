from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.gsc.layers import EncoderBlock, ConvBlock, DecoderBlock
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from typing import Tuple, List


class GSC(CNNModel):

    def __init__(self,
                 input_size: Tuple[int, int, int] = (256, 256, 1),
                 filters: List[int] = None):
        """
        Initializes the model which represents the GSC architecture.

        References:
            Promising crack segmentation method based on gated skip connection. M Jabreel and M Abdel-Nasser.
            Electronic Letters, Volume 56, Issue 10, 2020.

        Args:
            input_size: A tuple of two elements, representing the size of the input images. The GSC architecture only allows to handle grey-scale images.
            filters: A list of integers, representing the sizes of the consecutive filters to be applied. The first element of the list must be one,
                     since it is used for the final output.
        """
        super().__init__(input_size)

        if filters is None:
            filters = [1, 32, 64, 128, 256, 512, 1024]

        self.__filters = filters
        self.__history = None

    def build(self):
        """
        Builds the model and constructs the graph.
        """
        input_image = Input(self.input_size, name="input_image")

        y_history = []
        x = input_image

        for filter_size in self.__filters[1:len(self.__filters) - 1]:
            encoder = EncoderBlock(number_filters=filter_size)
            x, y_encoder = encoder(x)
            y_history.append(y_encoder)

        conv_block = ConvBlock(self.__filters[len(self.__filters) - 1], kSize=3)
        x = conv_block(x)
        y_history.reverse()

        for index, filter_size in enumerate(reversed(self.__filters[1:len(self.__filters) - 1])):
            decoder = DecoderBlock(number_filters=filter_size)
            x = decoder([x, y_history[index]])

        mask_out = Conv2D(filters=self.__filters[0], kernel_size=(1, 1), activation='sigmoid')(x)

        model = Model(inputs=input_image, outputs=mask_out, name='GSC')
        self.set_model(model)

        return input_image, mask_out
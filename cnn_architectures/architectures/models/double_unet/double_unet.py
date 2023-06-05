from tensorflow.keras.applications import *
from typing import Tuple

from cnn_architectures.architectures.base.CNNModel import CNNModel
from cnn_architectures.architectures.models.double_unet.layers import *
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.models import Model
import warnings


class DoubleUNet(CNNModel):

    def __init__(self, input_size: Tuple[int, int, int]):
        super().__init__(input_size)

        self.__history = None

        self.__skipped_connections_1 = []
        self.__skipped_connections_2 = []
        self.__vgg = None

    def build(self):
        input_image = keras_layer.Input(self.input_size, name="input_image")

        self.__vgg = VGG19(include_top=False, weights='imagenet', input_tensor=input_image)

        x = self.__vgg.get_layer("block5_conv4").output
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        for name in names:
            self.__skipped_connections_1.append(self.__vgg.get_layer(name).output)
        x = ASPP(filters=64, input_shape=x.shape)(x)

        # x = ForwardConnectedDecoder(connections=self.__skipped_connections_1, input_size=x.shape)(x)
        #
        for idx, filter_size in enumerate([256, 128, 64, 32]):
            x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            x = concatenate([x, self.__skipped_connections_1[4 - (idx + 1)]])
            x = ConvolutionalBlock(filters = filter_size)(x)

        output1 = OutputBlock()(x)
        x = input_image * output1
        # x, skip2 = ForwardEncoder()(x)
        #
        for idx, filter_size in enumerate([32, 64, 128, 256]):
            x = ConvolutionalBlock(filters=filter_size)(x)
            self.__skipped_connections_2.append(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        x = ASPP(filters=64, input_shape=x.shape)(x)
        # x = ForwardDoubleConnectedDecoder(connections1=self.__skipped_connections_1, connections2=skip2)(x)
        #
        for idx, filter_size in enumerate([256, 128, 64, 32]):
            x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            x = concatenate([x, self.__skipped_connections_1[4 - (idx + 1)], self.__skipped_connections_2[4 - (idx + 1)]])
            x = ConvolutionalBlock(filters=filter_size)(x)

        output2 = OutputBlock()(x)
        mask_out = concatenate([output1, output2], name='mask_out')
        model = Model(inputs=input_image, outputs=mask_out)
        self.set_model(model)

        return input_image, mask_out


if __name__=='__main__':
    from cnn_architectures.utils.dataset_generator import get_dataset
    from cnn_architectures.utils.metrics import dice_coef, dice_loss
    import os
    from tensorflow.keras.callbacks import CSVLogger
    from cnn_architectures.utils.callbacks import WeightsSaver

    model = DoubleUNet(input_size=(256, 256, 3))
    model.build()
    model.compile(learning_rate=0.0001,
                  loss_func=dice_loss,
                  metrics=[dice_coef])

    model_name='DoubleUNet'
    batch_size = 4
    num_epochs = 10

    train_image_dir = 'C:/Users/tonin/Desktop/Mates/TFG/TFG_Antonio_Nadal/database_v3/train/org/'
    train_mask_dir = 'C:/Users/tonin/Desktop/Mates/TFG/TFG_Antonio_Nadal/database_v3/train/mask/'

    test_image_dir = 'C:/Users/tonin/Desktop/Mates/TFG/TFG_Antonio_Nadal/database_v3/test/org/'
    test_mask_dir = 'C:/Users/tonin/Desktop/Mates/TFG/TFG_Antonio_Nadal/database_v3/test/mask/'

    weights_save_path = f'./{model_name}_train/'
    log_save_path = f'./{model_name}_train/'

    try:
        os.makedirs(weights_save_path)
    except:
        pass

    callbacks = [
        WeightsSaver(save_freq=1, save_path=weights_save_path),
        CSVLogger(f'{log_save_path}/train_log.csv')
    ]

    train_dataset = get_dataset(train_image_dir,
                                train_mask_dir,
                                batch_size=batch_size,
                                num_epochs=num_epochs,
                                mask_size=(256, 256, 28))
    test_dataset = get_dataset(test_image_dir,
                               test_mask_dir,
                               batch_size=batch_size,
                               num_epochs=num_epochs,
                               mask_size=(256, 256, 28))

    model.train(train_generator=train_dataset,
                val_generator=test_dataset,
                val_steps=150,
                steps_per_epoch=601,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose=1,
                )
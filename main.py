from models.double_unet.double_unet import DoubleUNet

if __name__ == '__main__':

    model1 = DoubleUNet(input_size=(256, 256, 3))
    model1.build()
    model1.compile()

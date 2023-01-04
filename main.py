from architectures.models.gsc.gsc import GSC

if __name__ == '__main__':

    """
    model = DoubleUNet(input_size=(256, 256, 3))
    model.build()
    model.compile()
    """

    """
    """
    model = GSC(input_size=(256, 256, 1),
                filters=[1, 32, 64, 128, 256, 512, 1024])
    model.build()
    model.compile()

    print(model.model.summary())
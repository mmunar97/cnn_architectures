from tensorflow.keras.callbacks import Callback


class WeightsSaver(Callback):
    def __init__(self, save_freq: int, save_path: str):
        super(WeightsSaver, self).__init__()
        self.freq = save_freq
        self.save_path = save_path
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch % self.freq == 0:
            self.model.save_weights(f'{self.save_path}/epoch_{self.epoch+1}.hdf5')
        self.epoch += 1

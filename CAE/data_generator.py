import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, shuffle=False, batch_size=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_dir=data_dir
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.batch_size is None:
            return int(np.floor(len(self.list_IDs)))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.batch_size is None:
            indexes = self.indexes[index]
            list_IDs_temp = [self.list_IDs[index]]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        if self.batch_size is None:
            X = np.load(f'{self.data_dir}{list_IDs_temp[0]}.npy')
        else:
            X = []
            for i, ID in enumerate(list_IDs_temp):
                patch = np.load(f'{self.data_dir}{ID}.npy')
                X.append(patch)
            X = np.array(X)
        return X
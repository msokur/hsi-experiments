import os
import glob
import numpy as np
from tensorflow import keras

import config


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    # modes: 'train', 'valid', 'all'
    def __init__(self, mode,
                 shuffled_npz_path,
                 batches_npz_path,
                 log_dir,
                 except_indexes=[],
                 batch_size=config.BATCH_SIZE,
                 split_factor=config.SPLIT_FACTOR,
                 split_flag=True,
                 for_tuning=False):
        self.class_weight = None
        import preprocessor

        '''Initialization'''
        self.raw_npz_path = config.RAW_NPZ_PATH
        self.shuffled_npz_path = shuffled_npz_path
        self.batches_npz_path = batches_npz_path
        self.mode = mode
        self.split_factor = split_factor
        self.log_dir = log_dir
        self.split_flag = split_flag
        self.for_tuning = for_tuning

        self.shuffled_npz_paths = glob.glob(os.path.join(shuffled_npz_path, 'shuffl*.npz'))

        self.batch_size = batch_size
        self.except_indexes = except_indexes
        self.index = 0

        self.preprocessor = preprocessor.Preprocessor()

        print('--------------------PARAMS----------------------')
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))
        print('------------------------------------------------')

        self.split()
        self.len = self.__len__()
        print('self.len', self.len)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.batches_npz_path)

    def __getitem__(self, index):
        """Generate one batch of data"""
        data = np.load(self.batches_npz_path[index])
        X, y = data['X'], data['y']

        if config.WITH_SAMPLE_WEIGHTS and 'weights' in data.keys():
            return X, y, data['weights']
        # print(X.shape, y.shape)

        return X, y.astype(np.float)

    ''' #Self-test public method - the copy of private __getitem__'''

    def getitem(self, index):
        return self.__getitem__(index)

    def split(self, except_indexes=None):
        if except_indexes is None:
            except_indexes = self.except_indexes
        else:
            self.except_indexes = except_indexes

        if self.for_tuning:
            self.shuffled_npz_paths = self.shuffled_npz_paths[:1]

        if self.split_flag:
            self.preprocessor.split_data_into_npz_of_batch_size(self.shuffled_npz_paths,
                                                                self.batch_size,
                                                                self.batches_npz_path,
                                                                self.log_dir,
                                                                self.except_indexes)
        else:
            print('!!!!!   Dataset not split   !!!!!')

        batches_paths = glob.glob(os.path.join(self.batches_npz_path, '*.npz'))  # TODO, for test, remove!!!

        if self.for_tuning:
            batches_paths = batches_paths[:10]

        split_factor = int(self.split_factor * len(batches_paths))

        if self.mode == 'all':
            self.batches_npz_path = batches_paths.copy()
        if self.mode == 'train':
            self.batches_npz_path = batches_paths[:split_factor]
        if self.mode == 'valid':
            self.batches_npz_path = batches_paths[split_factor:]

    def get_class_weights(self, labels=None):
        if labels is None:
            labels = config.LABELS_OF_CLASSES_TO_TRAIN
        labels = np.array(labels)
        sums = np.zeros(labels.shape)

        for p in self.batches_npz_path:
            data = np.load(p)
            y = data['y']
            for i, l in enumerate(labels):
                sums[i] += np.flatnonzero(y == l).shape[0]

        total = np.sum(sums)

        # weight_for_0 = (1 / neg)*(total)/2.0
        # weight_for_1 = (1 / pos)*(total)/2.0

        weights = {}
        for i, l in enumerate(labels):
            weights[l] = (1 / sums[i]) * total / 2.0

        self.class_weight = weights

        return self.class_weight

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.index = 0


if __name__ == '__main__':
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    # sys.path.insert(1, os.path.join(parentdir, 'utils'))
    import config

    dataGenerator = DataGenerator('train', '/work/users/mi186veva/data_preprocessed/augmented',
                                  '/work/users/mi186veva/data_preprocessed/augmented/shuffled',
                                  '/work/users/mi186veva/data_preprocessed/augmented/batch_sized', '', split_flag=False)
    print(len(dataGenerator.batches_npz_path))
    print(dataGenerator.get_class_weights(labels=[0, 1]))
    X_, y_ = dataGenerator.getitem(
        0)  # Marianne, if you want to try this line - you need to unсomment the public 'getitem' method

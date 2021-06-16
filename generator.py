import os
import glob
import numpy as np
from preprocessor import Preprocessor
import config
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, raw_npz_path, shuffled_npz_path, splitted_npz_path,  except_indexes=[], batch_size=config.BATCH_SIZE):
        '''Initialization'''
        self.raw_npz_path = raw_npz_path
        self.shuffled_npz_path = shuffled_npz_path
        self.splitted_npz_path = splitted_npz_path
        
        self.raw_npz_paths = glob.glob(os.path.join(raw_npz_path, '*.npz'))
        self.shuffled_npz_paths = glob.glob(os.path.join(shuffled_npz_path, 'shuffl*.npz'))
        
        self.batch_size = batch_size
        self.except_indexes = except_indexes
        self.index = 0
        
        self.preprocessor = Preprocessor()
        self.split()


    def __len__(self):
        '''Denotes the number of batches per epoch'''
        
        return len(glob.glob(os.path.join(self.splitted_npz_path), ".npz"))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        data = np.load(self.splitted_npz_paths[index])
        X, y = data['X'], data['y']

        self.index += 1
        
        return X, y
    
    '''def getitem(self, index):
        data = np.load(self.splitted_npz_paths[index])
        X, y = data['X'], data['y']

        self.index += 1
        
        return X, y'''
    
    def split(self, except_indexes=None):
        if except_indexes is None:
            except_indexes = self.except_indexes
        else:
            self.except_indexes = except_indexes
            
        self.preprocessor.split_data_into_npz_of_batch_size(self.shuffled_npz_paths, self.batch_size, self.splitted_npz_path, self.except_indexes)
        self.splitted_npz_paths = glob.glob(os.path.join(self.splitted_npz_path, '*.npz'))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = 0
        
if __name__ == '__main__':
    dataGenerator = DataGenerator('/work/users/mi186veva/data_preprocessed/augmented', 
                                 '/work/users/mi186veva/data_preprocessed/augmented/shuffled',
                                 '/work/users/mi186veva/data_preprocessed/augmented/batch_sized')
    X, y = dataGenerator.getitem(0)

import os
import glob
import numpy as np
from preprocessor import Preprocessor
import config
from tensorflow import keras
from data_loader import save_scaler, restore_scaler
from sklearn import preprocessing
from scipy.signal import savgol_filter


class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    
    #modes: 'train', 'valid', 'all'
    def __init__(self, mode, 
                 raw_npz_path, 
                 shuffled_npz_path, 
                 splitted_npz_path, 
                 log_dir, 
                 except_indexes=[], 
                 batch_size=config.BATCH_SIZE, 
                 split_factor=config.SPLIT_FACTOR, 
                 split_flag=True):
        '''Initialization'''
        self.raw_npz_path = raw_npz_path
        self.shuffled_npz_path = shuffled_npz_path
        self.splitted_npz_path = splitted_npz_path
        self.mode = mode
        self.split_factor = split_factor
        self.log_dir = log_dir
        self.split_flag = split_flag
        
        self.raw_npz_paths = glob.glob(os.path.join(raw_npz_path, '*.npz'))
        self.shuffled_npz_paths = glob.glob(os.path.join(shuffled_npz_path, 'shuffl*.npz'))
        
        self.batch_size = batch_size
        self.except_indexes = except_indexes
        self.index = 0
        
        self.preprocessor = Preprocessor()
        self.split()


    def __len__(self):
        '''Denotes the number of batches per epoch'''
        
        return len(self.splitted_npz_paths)

    def __getitem__(self, index, flag=False):
        '''Generate one batch of data'''
        data = np.load(self.splitted_npz_paths[index])
        X, y = data['X'], data['y']
        
        '''if len(X.shape) == 3:  #TODO Marianne, I think you will need to rewrite it (just take X as it is without tis block)
            y = [ [_y_] * X.shape[1] for _y_ in y]
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)'''
        
        #Marianne, about normalizers - I will implement SNV later, for now make your data normalized before shuffle etc.
        #X = preprocessing.Normalizer().transform(X[:, :-1])    #TODO, Marianne, du brauchst nicht [:, :-1] in X machen
        
       
        #X = savgol_filter(X, 5, 2)
        
        #X = X[:, :-1]
        self.index += 1
        
        return X, y
    
    ''' #Self-test public method - the copy of private __getitem__
    def getitem(self, index):
        data = np.load(self.splitted_npz_paths[index])
        X, y = data['X'], data['y']
        
        if len(X.shape) == 3:  #TODO Marianne, I think you will need to rewrite it (just take X as it is without tis block)
            y = [ [_y_] * X.shape[1] for _y_ in y]
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
        
        X = X[:, :-1]    #TODO, Marianne, du brauchst nicht [:, :-1] in X machen
        self.index += 1
        
        return X, y'''
    
    def split(self, except_indexes=None):
        if except_indexes is None:
            except_indexes = self.except_indexes
        else:
            self.except_indexes = except_indexes
        
        if self.split_flag:
            self.preprocessor.split_data_into_npz_of_batch_size(self.shuffled_npz_paths, self.batch_size, self.splitted_npz_path, self.log_dir, self.except_indexes)
        else:
            print('!!!!!   Dataset not splitted   !!!!!')
            
        splitted_paths = glob.glob(os.path.join(self.splitted_npz_path, '*.npz'))
        split_factor = int(self.split_factor * len(splitted_paths))
        
        if self.mode == 'all':
            self.splitted_npz_paths = splitted_paths.copy()
        if self.mode == 'train':
            self.splitted_npz_paths = splitted_paths[:split_factor]
        if self.mode == 'valid':
            self.splitted_npz_paths = splitted_paths[split_factor:]

    def get_class_weights(self):
        summ_ill = 0
        summ_healthy = 0
        
        for p in self.splitted_npz_paths:
            data = np.load(p)
            summ_ill += np.where(data['y'] == 1)[0].shape[0]
            summ_healthy += np.where(data['y'] == 0)[0].shape[0]
        
        neg = summ_healthy
        pos = summ_ill
        total = neg + pos

        weight_for_0 = (1 / neg)*(total)/2.0 
        weight_for_1 = (1 / pos)*(total)/2.0

        self.class_weight = {0: weight_for_0, 1: weight_for_1}
        
        return self.class_weight
            
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = 0
        
if __name__ == '__main__':
    dataGenerator = DataGenerator('valid', '/work/users/mi186veva/data_preprocessed/augmented', 
                                 '/work/users/mi186veva/data_preprocessed/augmented/shuffled',
                                 '/work/users/mi186veva/data_preprocessed/augmented/batch_sized', '')
    print(len(dataGenerator.splitted_npz_paths))
    print(dataGenerator.get_class_weights())
    X, y = dataGenerator.getitem(0)  #Marianne, if you want to try this line - you need to unkomment the public 'getitem' method

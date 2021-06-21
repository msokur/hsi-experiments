import os
import numpy as np
import random
import glob
from tqdm import tqdm
import math
import config
import data_loader
from sklearn import preprocessing
'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''



class Preprocessor():
    def __init__(self, load_name_for_x='X', load_name_for_y='y'):  #Marianne, I think you need to change load_name_for_x
        self.dict_names = [load_name_for_x, load_name_for_y, 'PatientName', 'PatientIndex'] 
    
    #------------------divide all samples into piles_number files------------------
    def __create_piles(self):
        print('----Piles creating started----')
        
        piles = []
        for i in range(self.piles_number):
            piles.append({})
            for d_name in self.dict_names:
                piles[i][d_name] = []
         
        
        print('--Splitting into piles started--')
        for i, p in tqdm(enumerate(self.shuffle_paths)):
            name = p.split("/")[-1].split(".")[0]
            data = np.load(p)
            X, y = data[self.dict_names[0]], data[self.dict_names[1]]

            for _X, _y in zip(X, y):
                pile = random.randint(0, self.piles_number - 1)
                
                values = [_X, _y, name, i]
                
                for it, d_name in enumerate(self.dict_names):
                    piles[pile][d_name].append(values[it])
        
        print('--Splitting into piles finished--')
        for i, pile in enumerate(piles):
            np.savez(os.path.join(self.shuffle_saving_path, 'pile'+str(i)), **{n: a for n, a in pile.items()})
        
        print('----Piles creating finished----')
            
    def __shuffle_piles(self):
        print('----Shuffling of piles started----')
        piles_paths = glob.glob(os.path.join(self.shuffle_saving_path, 'pile*.npz'))
        
        for i, pp in enumerate(piles_paths):
            data = np.load(pp)
            
            indexes = np.arange(data[self.dict_names[0]].shape[0])
            random.shuffle(indexes)
            
            os.remove(pp)
            np.savez(os.path.join(self.shuffle_saving_path, 'shuffled'+str(i)), **{n: a[indexes] for n, a in data.items()})
        
        print('----Shuffling of piles finished----')
    
    def __split_arrays(self, X, y, p_names, p_indexes):
         #---------------splitting into archives----------
        chunks = X.shape[0] // self.batch_size
        chunks_max = chunks * self.batch_size
        X_arr = np.array_split(X[:chunks_max], chunks)
        y_arr = np.array_split(y[:chunks_max], chunks)
        names_arr = np.array_split(p_names[:chunks_max], chunks)
        indexes_arr = np.array_split(p_indexes[:chunks_max], chunks)

        #---------------saving of the non equal last part for the future partition---------
        self.rest_X.append(X[chunks_max:])
        self.rest_y.append(y[chunks_max:])
        self.rest_names.append(p_names[chunks_max:])
        self.rest_indexes.append(p_indexes[chunks_max:])

        #---------------saving of the non equal last part for the future partition---------
        ind = len(glob.glob(os.path.join(self.archives_of_batch_size_saving_path, "*")))
        for _X, _y, _n, _i in zip(X_arr, y_arr, names_arr, indexes_arr):
            arch = {}
            values = [_X, _y, _n, _i]
            for i, n in enumerate(self.dict_names):
                arch[n] = values[i]

            np.savez(os.path.join(self.archives_of_batch_size_saving_path, 'batch'+str(ind)), **{n: a for n, a in arch.items()})
            ind+=1
        
    def shuffle(self, paths, piles_number, shuffle_saving_path):
        print('--------Shuffling started--------')
        self.shuffle_paths = paths
        self.piles_number = piles_number
        self.shuffle_saving_path = shuffle_saving_path
        
        if not os.path.exists(self.shuffle_saving_path):
                os.mkdir(self.shuffle_saving_path)
        
        self.__create_piles()
        self.__shuffle_piles()
        print('--------Shuffling finished--------')
        
        
                
    def split_data_into_npz_of_batch_size(self, paths, batch_size, archives_of_batch_size_saving_path, scaler_saving_path, except_names=[], not_certain=config.NOT_CERTAIN_FLAG):
        print('--------Splitting into npz of batch size started--------')
        self.batch_size = batch_size
        self.archives_of_batch_size_saving_path = archives_of_batch_size_saving_path
        
        if not os.path.exists(archives_of_batch_size_saving_path):
                os.mkdir(archives_of_batch_size_saving_path)
                
        #------------removing of previously generated archives (of the previous CV step) ----------------
        files = glob.glob(os.path.join(archives_of_batch_size_saving_path, '*.npz'))
        for f in files:
            os.remove(f)
                    
        self.rest_X, self.rest_y, self.rest_names, self.rest_indexes = [], [], [], []
        for p in paths:
            #------------ except_indexes filtering ---------------
            data = np.load(p)
            X, y, p_names, p_indexes = data[self.dict_names[0]], \
                                       data[self.dict_names[1]], \
                                       data[self.dict_names[2]], \
                                       data[self.dict_names[3]]
            indexes = np.arange(X.shape[0])
            for except_name in except_names:
                indexes = np.flatnonzero(np.core.defchararray.find(p_names, except_name) == -1)
                X, y, p_names, p_indexes = X[indexes], y[indexes], p_names[indexes], p_indexes[indexes]
            
            if not not_certain:
                indexes = np.flatnonzero(y != 2)
                X, y, p_names, p_indexes = X[indexes], y[indexes], p_names[indexes], p_indexes[indexes]
            
            self.__split_arrays(X, y, p_names, p_indexes)  
        
        #------------------save rest of rest archives----------------
        rest_X, rest_y, rest_names, rest_indexes = np.array(self.rest_X), np.array(self.rest_y), np.array(self.rest_names), np.array(self.rest_indexes)
        self.rest_X, self.rest_y, self.rest_names, self.rest_indexes = [], [], [], []
        
        rest_X = np.concatenate(rest_X, axis=0)
        rest_y = np.concatenate(rest_y, axis=0)
        rest_names = np.concatenate(rest_names, axis=0)
        rest_indexes = np.concatenate(rest_indexes, axis=0)
        
        if rest_X.shape[0] >= batch_size:
            self.__split_arrays(rest_X, rest_y, rest_names, rest_indexes) 
        
        print('--------Splitting into npz of batch size finished--------')
        

if __name__ == '__main__':
    preprocessor = Preprocessor()
    paths = glob.glob('/work/users/mi186veva/data_preprocessed/augmented/*.npz')
    print(len(paths))
    #preprocessor.shuffle(['/work/users/mi186veva/data_preprocessed/augmented/2019_07_12_11_15_49_.npz', '/work/users/mi186veva/data_preprocessed/augmented/2020_03_27_16_56_41_.npz'], 20, '/work/users/mi186veva/data_preprocessed/augmented/shuffled')
    #preprocessor.shuffle(paths, 100, '/work/users/mi186veva/data_preprocessed/augmented/shuffled')
    preprocessor.split_data_into_npz_of_batch_size(['/work/users/mi186veva/data_preprocessed/augmented/shuffled/shuffled9.npz', '/work/users/mi186veva/data_preprocessed/augmented/shuffled/shuffled8.npz'], 64, '/work/users/mi186veva/data_preprocessed/augmented/batch_sized', except_names=['2020_03_27_16_56_41', '2020_05_15_12_43_58'])
        
        
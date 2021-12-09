import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, os.path.join(parentdir, 'utils')) 


import os
import numpy as np
import random
import glob
from tqdm import tqdm
import math
from sklearn import preprocessing
import pickle
from scipy.signal import savgol_filter
from sklearn import preprocessing
import time   #TODO remove
#from pathos.multiprocessing import ProcessingPool as Pool

import data_loader
from background_detection import detect_background
import config



'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''



class Preprocessor():
    def __init__(self, load_name_for_x='X', load_name_for_y='y', piles_number=100):  #Marianne, I think you need to change load_name_for_x
        self.dict_names = [load_name_for_x, load_name_for_y, 'PatientName', 'PatientIndex'] 
        self.piles_number = piles_number
    
    #------------------divide all samples into piles_number files------------------
    def __create_piles(self):
        print('----Piles creating started----')
        
        #remove previous piles if they exist
        piles_paths = glob.glob(os.path.join(self.shuffle_saving_path, '*pile*'))
        for p in piles_paths:
            os.remove(p)
        
        #create clear piles
        piles = []
        for i in range(self.piles_number):
            piles.append([])
            open(os.path.join(self.shuffle_saving_path, str(i)+'.pile'), 'w').close() #creating of an empty file
        
        print('--Splitting into piles started--')
        
        for i, p in tqdm(enumerate(self.raw_paths)):
            #clear piles for new randon numbers
            for pn in range(self.piles_number):
                piles[pn] = []
            
            name = p.split("/")[-1].split(".")[0]
            _data = np.load(p)
            
            data = {n: a for n, a in _data.items()}
            X, y = data[self.dict_names[0]], data[self.dict_names[1]]
            
            bg_mask = np.ones(X.shape[0]).astype(np.bool)
            if config.WITH_BACKGROUND_EXTRACTION:
                bg_mask = self.background_extraction(X)
            
            if self.augmented:  
                y = [ [_y_] * X.shape[1] for _y_ in y ]
                data[self.dict_names[0]] = np.concatenate(X, axis=0)
                data[self.dict_names[1]] = np.concatenate(y, axis=0)
            
            data = {n: a[bg_mask] for n, a in data.items()}
            
            #fill random distribution to files
            for it in range(data[self.dict_names[0]].shape[0]):
                pile = random.randint(0, self.piles_number - 1)
                piles[pile].append(it)            
            
            for i_pile, pile in enumerate(piles):              
                _names = [name] * len(pile)
                _indexes = [i] * len(pile)
                                
                values = [data[self.dict_names[0]][pile], data[self.dict_names[1]][pile], _names, _indexes]
                _values = {k: v for k, v in zip(self.dict_names, values)}
                pickle.dump(_values, open(os.path.join(self.shuffle_saving_path, str(i_pile)+'.pile'), 'ab'))
                    
        print('--Splitting into piles finished--')
        
        print('----Piles creating finished----')
            
    def __shuffle_piles(self):
        print('----Shuffling of piles started----')
        piles_paths = glob.glob(os.path.join(self.shuffle_saving_path, '*.pile'))
        print(len(piles_paths))
        
        for i, pp in tqdm(enumerate(piles_paths)):
            data = []
            with open(pp, 'rb') as fr:
                try:
                    while True:
                        data.append(pickle.load(fr))
                except EOFError:
                    pass            
            
            _data = {}
            for key in data[0].keys():
                _data[key] = [f[key] for f in data]
                _data[key] = np.concatenate(_data[key], axis=0)

            indexes = np.arange(_data[self.dict_names[0]].shape[0])
            random.shuffle(indexes)
            
            os.remove(pp)
            np.savez(os.path.join(self.shuffle_saving_path, 'shuffled'+str(i)), **{n: a[indexes] for n, a in _data.items()})
        
        print('----Shuffling of piles finished----')
    
    
    def background_extraction(self, X):
        
        bg_mask = detect_background(X)
        bg_mask = bg_mask == 1
        
        return bg_mask
        
    
    '''Add here some specific preprocessing if needed'''
    def preprocess(self, X, y):
        #ill_indexes = np.flatnonzero(_y == 1)   #TODO remove!
        #_X[ill_indexes, :-1] = savgol_filter(_X[ill_indexes, :-1], 5, 2) #TODO remove!
        
        #X[X < 0] = 0.
        #X = preprocessing.Normalizer().transform(X[:, :-1]) #TODO be careful
        
        ill_indexes = np.flatnonzero(y == 1) 
        
        #X = X[:, :-1]
        X[X<0] = 0.
        #X[ill_indexes] = savgol_filter(X[ill_indexes], 7, 2)
        X[X<0] = 0.
        #X = preprocessing.Normalizer().transform(X)
        #X = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    
        return X
    
    def shuffle(self, paths, piles_number, shuffle_saving_path, augmented=False):
        print('--------Shuffling started--------')
        self.raw_paths = paths
        self.piles_number = piles_number
        self.shuffle_saving_path = shuffle_saving_path
        self.augmented = augmented
        
        if not os.path.exists(self.shuffle_saving_path):
             os.mkdir(self.shuffle_saving_path)
        
        self.__create_piles()
        self.__shuffle_piles()
        print('--------Shuffling finished--------')
        
    
    def X_y_concatenate(self, root_path):
        paths = glob.glob(os.path.join(root_path, '*.npz'))
        
        X, y, indexes = [], [], []
        for path in paths:
            data = np.load(path)
            _X, _y, _i = data['X'], data['y'], data['indexes_in_datacube']
            
            X += list(_X)
            y += list(_y)
            indexes += list(_i)
            
        X, y, indexes = np.array(X), np.array(y), np.array(indexes)
        
        return X, y, indexes
    
    def scaler_save(self, scaler, scaler_path):
        pickle.dump(scaler, open(scaler_path, 'wb'))

    def scaler_restore(self, scaler_path):
        return pickle.load(open(scaler_path, 'rb'))
    
    def fit_scale_from_path(self, root_path, scaler_saving_path, scaling_type=config.NORMALIZATION_TYPE):
        X, _, _ = self.X_y_concatenate(root_path)
        return self.scale_from_X(X, scaler_saving_path, scaling_type=scaling_type)
    
    def fit_scale_from_X(self, X, scaler_saving_path, scaling_type=config.NORMALIZATION_TYPE):        
        scaler = None
        
        if scaling_type == 0: #svn
            scaler = preprocessing.StandartScaler().fit(X)
        else: #l2_norm
            scaler = preprocessing.Normalizer().fit(X)
        
        self.scaler_save(scaler, scaler_saving_path)
        return scaler.transform(X)
    
    def scale_X(self, X, scaler_path):
        _3d = False
        shapes = []
        
        if X.shape[0] != 0:
            scaler = self.scaler_restore(scaler_path)
            print(len(X.shape))
            if len(X.shape) > 2:
                _3d = True
                shapes = X.shape
                X = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1]))

            X = scaler.transform(X)

            if _3d:
                X = np.reshape(X, shapes)
        
        return X
            
            
    
    def scaledData_save(self, root_path, destination_path, scaler_path):
        paths = glob.glob(os.path.join(root_path, '*.npz'))
        scaler = self.scaler_restore(scaler_path)
        
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        
        for path in paths:
            data = np.load(path)
            X = data['X']
            X = self.scale_X(X, scaler_path)
            data = {n: a for n, a in data.items()}
            data['X'] = X.copy()
            np.savez(os.path.join(destination_path, data_loader.DataLoader.name_get(path)), **data)
        
    
    def __split_arrays(self, X, y, p_names, p_indexes):
             #---------------splitting into archives----------
            chunks = X.shape[0] // self.batch_size
            chunks_max = chunks * self.batch_size
            X_arr = np.array_split(X[:chunks_max], chunks)
            y_arr = np.array_split(y[:chunks_max], chunks)
            names_arr = np.array_split(p_names[:chunks_max], chunks)
            indexes_arr = np.array_split(p_indexes[:chunks_max], chunks)

            #---------------saving of the non equal last part for the future partition---------
            self.rest_X += list(X[chunks_max:])
            self.rest_y += list(y[chunks_max:])
            self.rest_names += list(p_names[chunks_max:])
            self.rest_indexes += list(p_indexes[chunks_max:])

            #---------------saving of the non equal last part for the future partition---------
            ind = len(glob.glob(os.path.join(self.archives_of_batch_size_saving_path, "*")))
            for _X, _y, _n, _i in zip(X_arr, y_arr, names_arr, indexes_arr):
                arch = {}

                _X = self.preprocess(_X, _y) 

                values = [_X, _y, _n, _i]
                for i, n in enumerate(self.dict_names):
                    arch[n] = values[i]

                np.savez(os.path.join(self.archives_of_batch_size_saving_path, 'batch'+str(ind)), **arch)
                ind += 1         
                
    def split_data_into_npz_of_batch_size(self, paths, batch_size, archives_of_batch_size_saving_path, scaler_saving_path, except_names=[], not_certain=config.NOT_CERTAIN_FLAG):
        
        
        
        
        print('--------Splitting into npz of batch size started--------')
        self.batch_size = batch_size
        self.archives_of_batch_size_saving_path = archives_of_batch_size_saving_path
        
        if not os.path.exists(archives_of_batch_size_saving_path):
                os.mkdir(archives_of_batch_size_saving_path)
                
        for except_name in except_names:
                print(f'We except {except_name}')
                
        #------------removing of previously generated archives (of the previous CV step) ----------------
        files = glob.glob(os.path.join(archives_of_batch_size_saving_path, '*.npz'))
        for f in files:
            os.remove(f)
                    
        self.rest_X, self.rest_y, self.rest_names, self.rest_indexes = [], [], [], []
        for p in tqdm(paths):
            #------------ except_indexes filtering ---------------
            data = np.load(p)
            X, y, p_names, p_indexes = data[self.dict_names[0]], \
                                       data[self.dict_names[1]], \
                                       data[self.dict_names[2]], \
                                       data[self.dict_names[3]]
            indexes = np.arange(X.shape[0])
            for except_name in except_names:
                indexes = np.flatnonzero(np.core.defchararray.find(p_names, except_name) == -1)
                
                if indexes.shape[0] == 0:
                    print(f'WARNING! For except_name {except_name} no except_samples were found')
                
                X, y, p_names, p_indexes = X[indexes], y[indexes], p_names[indexes], p_indexes[indexes]
            
            if not not_certain:
                indexes = np.flatnonzero(y != 2)
                X, y, p_names, p_indexes = X[indexes], y[indexes], p_names[indexes], p_indexes[indexes]
            
            #p1 = Pool(4)
            #p1.map(__split_function, [X, y, p_names, p_indexes])
            self.__split_arrays(X, y, p_names, p_indexes)  
        
        #------------------save rest of rest archives----------------
        rest_X, rest_y, rest_names, rest_indexes = np.array(self.rest_X), np.array(self.rest_y), np.array(self.rest_names), np.array(self.rest_indexes)
        self.rest_X, self.rest_y, self.rest_names, self.rest_indexes = [], [], [], []
        
        #rest_X = np.concatenate(rest_X, axis=0)
        #rest_y = np.concatenate(rest_y, axis=0)
        #rest_names = np.concatenate(rest_names, axis=0)
        #rest_indexes = np.concatenate(rest_indexes, axis=0)
        
        rest_X = np.array(rest_X)
        rest_y = np.array(rest_y)
        rest_names = np.array(rest_names)
        rest_indexes = np.array(rest_indexes)
        
        if rest_X.shape[0] >= batch_size:
            #p2 = Pool(4)
            #p2.map(__split_function, [rest_X, rest_y, rest_names, rest_indexes])
            self.__split_arrays(rest_X, rest_y, rest_names, rest_indexes) 
        
        print('--------Splitting into npz of batch size finished--------')
        
    def pipeline(self, root_path, preprocessed_path, scaler_path=None, scaler_name='scaler'):
        #---------Data reading part--------------
        #dataLoader = data_loader.DataLoader(_3d=True, _3d_size=[5, 5])
        #dataLoader.datfiles_read_and_save_to_npz(root_path, preprocessed_path)
        
        #----------scaler part ------------------
        if scaler_path is None:
            scaler_path = os.path.join(preprocessed_path, scaler_name + config.SCALER_FILE_NAME)
            self.fit_scale_from_path(preprocessed_path, scaler_path)
        self.scaledData_save(preprocessed_path, preprocessed_path, scaler_path)
        
        #----------shuffle part------------------
        paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
        self.shuffle(paths, 
                     self.piles_number, 
                     os.path.join(preprocessed_path, 'shuffled'), 
                     augmented=False)
        

if __name__ == '__main__':
    preprocessor = Preprocessor()
    '''preprocessor.pipeline('/work/users/mi186veva/data', 
                          '/work/users/mi186veva/data_preprocessed/raw_3d', 
                         scaler_path='/work/users/mi186veva/data_preprocessed/raw/raw_all.scaler')'''
    paths = glob.glob('/work/users/mi186veva/data_preprocessed/raw_3d/*.npz')
    #print(len(paths))
    #preprocessor.shuffle(['/work/users/mi186veva/data_preprocessed/augmented/2019_07_12_11_15_49_.npz', '/work/users/mi186veva/data_preprocessed/augmented/2020_03_27_16_56_41_.npz'], 100, '/work/users/mi186veva/data_preprocessed/augmented_l2_norm/shuffled')
    #preprocessor.shuffle(paths, 100, '/work/users/mi186veva/data_preprocessed/raw_3d/shuffled', augmented=False)
    #preprocessor.shuffle(paths, 100, '/work/users/mi186veva/data_preprocessed/augmented/shuffled', augmented=True)
    
    #paths = glob.glob('/work/users/mi186veva/data_preprocessed/combi_with_raw_ill/shuffled/*.npz')
    preprocessor.split_data_into_npz_of_batch_size(paths, config.BATCH_SIZE, '/work/users/mi186veva/data_preprocessed/raw_3d/batch_sized1', "")
    
    
    #preprocessor.scale_from_path('/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw/raw_all.scaler')
    #start = time.time()
    #preprocessor.scaledData_save('/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw/raw_all.scaler')
    #end = time.time()
    #print('Time of scaledData_save', end - start)
        
        
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, os.path.join(parentdir, 'utils')) 

from sklearn.feature_extraction import image
import numpy as np
import cv2
from tqdm import tqdm
import glob
import time #TODO remove it
import scipy.io

from hypercube_data import Cube_Read
import config
import preprocessor
from background_detection import detect_background

class DataLoader():
    def __init__(self, dict_names=['X', 'y', 'indexes_in_datacube'], _3d=False, _3d_size=config._3D_SIZE):
        self.dict_names = dict_names
        self._3d = _3d
        self._3d_size = _3d_size
    
    def spectrum_read_from_dat(self, dat_path):  #+++++++++++++++++++++++++++
        spectrum_data, _ = Cube_Read(dat_path, 
                             wavearea=config.WAVE_AREA, 
                             Firstnm=config.FIRST_NM, 
                             Lastnm=config.LAST_NM).cube_matrix()
        return spectrum_data
    
    def labeled_spectrum_get_from_dat(self, dat_path, mask_path=None): #+++++++++++++++++++++++++++
        if mask_path is None:
            mask_path = dat_path + '_Mask JW Kolo.png'
            
        spectrum = self.spectrum_read_from_dat(dat_path)
        mask = self.mask_read(mask_path)
        healthy_indexes, ill_indexes, not_certain_indexes = self.indexes_get_bool_from_mask(mask)
        
        return spectrum[healthy_indexes], spectrum[ill_indexes], spectrum[not_certain_indexes]
    
    def labeled_spectrum_get_from_npz(self, npz_path):  #+++++++++++++++++++++
        data = np.load(npz_path)
        X, y = data['X'], data['y']
        
        healthy_spectrum, ill_spectrum, not_certain_spectrum = self.labeled_spectrum_get_from_X_y(X, y)
        
        return healthy_spectrum, ill_spectrum, not_certain_spectrum
    
    def labeled_spectrum_get_from_X_y(self, X, y):  #+++++++++++++++++++++++++++++
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]
        return healthy_spectrum, ill_spectrum, not_certain_spectrum
    
    @staticmethod
    def name_get(path):
        return path.split('/')[-1].split('.')[0]
    
    def mask_read(self, mask_path):  #++++++++++++++++++++++++++++
        mask = cv2.imread(mask_path)[..., ::-1]
        return mask
    
    def indexes_get_bool_from_mask(self, mask):  #+++++++++++++++++++++++++++++++++
        healthy_indexes = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255) #blue
        ill_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0) #yellow
        not_certain_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0) #red
        
        return healthy_indexes, ill_indexes, not_certain_indexes
    
    def indexes_get_bool_from_mat_mask(self, mask):  
        healthy_indexes = (mask == 2) | (mask == 3) 
        ill_indexes = (mask == 1) 
        not_certain_indexes = (mask == 0)
        
        return healthy_indexes, ill_indexes, not_certain_indexes
    
    def indexes_get_np_from_mask(self, mask):  #++++++++++++++++++++++++++++
        indexes = self.indexes_get_bool_from_mask(mask)
        
        healthy_indexes, ill_indexes, not_certain_indexes = self.indexes_get_np_from_bool_indexes(indexes)
        
        return healthy_indexes, ill_indexes, not_certain_indexes
    
    def indexes_get_np_from_bool_indexes(self, healthy_indexes, ill_indexes, not_certain_indexes):       #+++++++++++++++++++++  
        healthy_indexes = np.where(healthy_indexes)
        ill_indexes = np.where(ill_indexes)
        not_certain_indexes = np.where(not_certain_indexes)
        
        return healthy_indexes, ill_indexes, not_certain_indexes
    
    def X_y_concatenate_from_spectrum(self, spectrums, idxs, labels=None):   
        X, y, indexes_in_datacube = [], [], []
        if labels is None:
            labels = np.arange(len(spectrums))
        
        for spectrum, label, idx in zip(spectrums, labels, idxs):
            X += list(spectrum)
            y += [label]  * len(spectrum)
            indexes_in_datacube += list(np.array(idx).T)
            
        X, y, indexes_in_datacube = np.array(X), np.array(y), np.array(indexes_in_datacube)
        
        assert X.shape[0] == y.shape[0]
        assert y.shape[0] == indexes_in_datacube.shape[0]
        
        print(X.shape, y.shape, indexes_in_datacube.shape)
        
        return X, y, indexes_in_datacube
    
    def X_y_dict_save_to_npz(self, dat_path, destination_path, values, name=None):
        if name is None:
            name = dat_path.split('/')[-1].split('SpecCube')[0]
        np.savez(os.path.join(destination_path, name), **{n: a for n, a in values.items()}) 
    
    def X_cube_create(self, X, idx):
        return
    
    def patients_weights_save(self):
        return
    
    def datfile_read(self, dat_path):
        
        print(f'Reading {dat_path}')
        spectrum = self.spectrum_read_from_dat(dat_path)
        mask = self.mask_read(dat_path + '_Mask JW Kolo.png')
        
        background_mask = self.background_get_mask(spectrum, mask.shape[:2])

        
        if self._3d:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        indexes = self.indexes_get_bool_from_mask(mask)
        indexes = [i * background_mask for i in indexes]
        spectrums = [spectrum[indexes[0]], spectrum[indexes[1]], spectrum[indexes[2]]]
        indexes_np = self.indexes_get_np_from_bool_indexes(*indexes)

        values = self.X_y_concatenate_from_spectrum(spectrums, indexes_np)

        values = {n: v for n, v in zip(self.dict_names, values)}
        
        return values
    
    def background_get_mask(self, spectrum, shapes):
        background_mask = np.ones(shapes).astype(np.bool)
        if config.WITH_BACKGROUND_EXTRACTION:
            background_mask = detect_background(spectrum)
            background_mask = np.reshape(shapes)
            
        return background_mask
    
    def matfile_read(self, mat_path):
        
        print(f'Reading {mat_path}')
        data = scipy.io.loadmat(mat_path)
        spectrum, mask = data['cube'], data['gtMap']
        
        #mask = self.mask_read(dat_path + '_Mask JW Kolo.png')
        background_mask = self.background_get_mask(spectrum, mask.shape[:2])
        
        if self._3d:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        indexes = self.indexes_get_bool_from_mat_mask(mask)
        indexes = [i * background_mask for i in indexes]
        spectrums = [spectrum[indexes[0]], spectrum[indexes[1]], spectrum[indexes[2]]]
        indexes_np = self.indexes_get_np_from_bool_indexes(*indexes)

        values = self.X_y_concatenate_from_spectrum(spectrums, indexes_np)

        values = {n: v for n, v in zip(self.dict_names, values)}
        
        return values
    
    '''def datfiles_read(self, root_path, dict_names=['X', 'y', 'indexes_in_datacube'], _3d=False):
        
        dat_paths = glob.glob(os.path.join(root_path, '*.dat'))
        
        values_all = []

        for dat_path in tqdm(dat_paths):
            
            
        return dat_paths, values_all'''
            
    def datfiles_read_and_save_to_npz(self, root_path, destination_path):  
        print('----Saving of .npz archives is started----')
        
        dat_paths = glob.glob(os.path.join(root_path, '*.dat'))
        #dat_paths, values_all = self.datfiles_read(root_path, dict_names=dict_names)

        for dat_path in tqdm(dat_paths):
            values = self.datfile_read(dat_path)
            self.X_y_dict_save_to_npz(dat_path, destination_path, values)
            
        print('----Saving of .npz archives is over----')
    
    def matfiles_read_and_save_to_npz(self, root_path, destination_path):  
        print('----Saving of .npz archives is started----')
        
        mat_paths = glob.glob(os.path.join(root_path, '*.mat'))
        #dat_paths, values_all = self.datfiles_read(root_path, dict_names=dict_names)

        for mat_path in tqdm(mat_paths):
            values = self.matfile_read(mat_path)
            self.X_y_dict_save_to_npz(mat_path, destination_path, values, name=mat_path.split('/')[-1].split('.')[0])
            
        print('----Saving of .npz archives is over----')
        
            
    def patches3d_get_from_spectrum(self, spectrum):
        size = self._3d_size
        #Better not to use non even sizes
        pad = [int((s - 1) / 2) for s in size]
        if size[0] % 2 == 1 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 1 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1] + 1), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1] + 1), (0, 0)))

        patches = image.extract_patches_2d(spectrum_, tuple(size))  
        patches = np.reshape(patches, (spectrum.shape[0], spectrum.shape[1], size[0], size[1], patches.shape[-1]))

        return patches
    
    '''def patches3d_save_to_npz():
        
        _data_ = np.load('utils/for_patches.npz', allow_pickle=True)


        for spectrum, healthy_indexes, ill_indexes, y_, name in zip(_data_['spectrums'],_data_['healthy_indexes_all'], _data_['ill_indexes_all'], _data_['y_all'], _data_['names']):
            patches = get_patches(spectrum, size=[3, 3])
            print('Shape of patches:', patches.shape)

            healthy_spectrum = patches[healthy_indexes]
            ill_spectrum = patches[ill_indexes]
            X_3d = np.array(list(healthy_spectrum) + list(ill_spectrum))

            np.savez(os.path.join('data_3d_npz', name), X = X_3d, y = y_)

            i+=1


        X, y = get_X_y('data_3d_npz/*.npz')
        X, y = shuffle(X, y)
        train_X_3d, train_y_3d, test_X_3d, test_y_3d = split(X, y)

        print('train_X_3d and train_y_3d shapes: ', train_X_3d.shape, train_y_3d.shape)

        assert train_X.shape[0] == train_X_3d.shape[0]'''

if __name__ == '__main__':
    #dataLoader = DataLoader(_3d=True, _3d_size=[5, 5])
    dataLoader = DataLoader()
    dataLoader.datfiles_read_and_save_to_npz('/work/users/mi186veva/data', '/work/users/mi186veva/data_preprocessed/raw')
    
    #for key, value in values[0].items() :
    #    print (key)
    
    #for dat_path, value in zip(dat_paths, values):
    #    print(dat_path, value['X'].shape, value['y'].shape, value['indexes_in_datacube'].shape)
        
    
        
import sys
import os
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(2, os.path.join(currentdir, 'data_utils')) 


import config

from data_utils.hypercube_data import *
import numpy as np
import os
import glob
from data_utils.data_loaders.archive import data_loader_base as data_loader
import cv2
from tqdm import tqdm

'''
Existing checking steps
1. check_dat_npz -- check .dat files and raw .npz
2. check_raw_npz_and_aug -- check raw .npz and augmented .npz
3. -- check augmented .npz and shuffled .npz
4. -- check shuffled .npz and splitted .npz
5. -- check if splitted .npz have right resize in generators
6. -- check if no not_certai are in batches (for now)
'''


class Checker():
    
    def check_dat_npz(self, dat_source_paths, npz_path):
        print('------check_dat_npz------')
        print(f'Params: dat_source_paths {dat_source_paths},\n npz_path {npz_path}')
        print('-------------------------')
        
        self.dat_source_paths = dat_source_paths
        self.raw_npz_path = npz_path
        
        npz_paths = glob.glob(os.path.join(npz_path, '*.npz'))
        
        dat_paths = []
        for path_dir in dat_source_paths:
            dat_paths += glob.glob(os.path.join(path_dir, '*.dat'))
        
        for npz in npz_paths:           
            name = npz.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0]
            dat = dat_paths[np.flatnonzero(np.core.defchararray.find(dat_paths, name) != -1)[0]]  
            print(f'We are checking {dat} and {npz}')
            
            d_gesund, d_ill, d_not_certain = 0, 0, 0
            
            with open(dat, newline='') as filex:
                filename=filex.name
                
                spectrum_data, _ = Cube_Read(filename, wavearea=config.WAVE_AREA, Firstnm=config.FIRST_NM, Lastnm=config.LAST_NM).cube_matrix()

                mask = cv2.imread(glob.glob(filename + '*.png')[0])[..., ::-1]
                
                d_gesund, d_ill, d_not_certain = data_loader.get_masks(mask)
                d_gesund, d_ill, d_not_certain = d_gesund[0].shape[0], d_ill[0].shape[0], d_not_certain[0].shape[0]
                
            data = np.load(npz)
            n_gesund = data['X'][np.where(data['y'] == 0)].shape[0]
            n_ill = data['X'][np.where(data['y'] == 1)].shape[0]
            n_not_certain = data['X'][np.where(data['y'] == 2)].shape[0]
            
            print(f'Differences: healthy - {d_gesund - n_gesund}, ill - {d_ill - n_ill}, not_certain - {d_not_certain - n_not_certain}, all - {d_gesund + d_ill + d_not_certain - data["X"].shape[0]}')
            if d_gesund - n_gesund != 0:
                print(f'dat_healthy vs npz_healthy: {d_gesund} vs {n_gesund}')
            if d_ill - n_ill != 0:
                print(f'dat_ill vs npz_ill: {d_ill} vs {n_ill}')
            if d_not_certain - n_not_certain != 0:
                print(f'dat_not_certain vs npz_not_certain: {d_not_certain} vs {n_not_certain}')
            
            print('----------------------------------------')
          
    def check_raw_npz_and_aug(self, raw_npz_path, aug_path):
        print('------check_raw_npz_and_aug------')
        print(f'Params: raw_npz_path {raw_npz_path},\n aug_path {aug_path}')
        print('-------------------------')
        
        raw_npz_paths = sorted(glob.glob(os.path.join(raw_npz_path, '*.npz')))
        aug_paths = sorted(glob.glob(os.path.join(aug_path, '*.npz')))
        
        for raw, aug in zip(raw_npz_paths, aug_paths):
            print(f'We are checking {raw} and {aug}')
            
            r_data = np.load(raw)
            r_gesund = r_data['X'][np.where(r_data['y'] == 0)].shape[0]
            r_ill = r_data['X'][np.where(r_data['y'] == 1)].shape[0]
            r_not_certain = r_data['X'][np.where(r_data['y'] == 2)].shape[0]
            
            a_data = np.load(aug)
            a_gesund = a_data['X'][np.where(a_data['y'] == 0)].shape[0]
            a_ill = a_data['X'][np.where(a_data['y'] == 1)].shape[0]
            a_not_certain = a_data['X'][np.where(a_data['y'] == 2)].shape[0]
            
            print(f'Differences: healthy - {r_gesund - a_gesund}, ill - {r_ill - a_ill}, not_certain - {r_not_certain - a_not_certain}, all - {r_data["X"].shape[0] - a_data["X"].shape[0]}')
            
            print('----------------------------------------')
            
    def check_source_and_shuffled(self, source_path, shuffled_path):
        print('------check_source_and_shuffled------')
        print(f'Params: source_path {source_path},\n shuffled_path {shuffled_path}')
        print('-------------------------')
        
        source_paths = glob.glob(os.path.join(source_path, '*.npz'))
        shuffled_paths = glob.glob(os.path.join(shuffled_path, 'shuffle*.npz'))
                
        for s in tqdm(source_paths):
            name = s.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0]
            print(f'We are checking {name}')
            
            s_data = np.load(s)
            s_gesund = s_data['X'][s_data['y'] == 0].shape[0]
            s_ill = s_data['X'][s_data['y'] == 1].shape[0]
            s_not_certain = s_data['X'][s_data['y'] == 2].shape[0]
            
            sh_all, sh_gesund, sh_ill, sh_not_certain = 0, 0, 0, 0
            for shuff in shuffled_paths:
                sh_data = np.load(shuff)
                
                #indx = np.flatnonzero(np.core.defchararray.find(sh_data['PatientName'], name) != -1)
                indx = np.flatnonzero(sh_data['PatientName'] == name)
                sh_all += indx.shape[0]
                sh_gesund += np.flatnonzero(sh_data['y'][indx] == 0).shape[0]
                sh_ill += np.flatnonzero(sh_data['y'][indx] == 1).shape[0]
                sh_not_certain += np.flatnonzero(sh_data['y'][indx] == 2).shape[0]
                
            print(f'Differences: healthy - {s_gesund - sh_gesund}, ill - {s_ill - sh_ill}, not_certain - {s_not_certain - sh_not_certain}, all - {s_data["X"].shape[0] - sh_all}')
                
            print('----------------------------------------')
            
    def check_source_and_batched(self, source_path, batched_path, preffix=''):
        print('------check_source_and_batched------')
        print(f'Params: source_path {source_path},\n batched_path {batched_path}, preffix {preffix}')
        print('-------------------------')
        
        source_paths = glob.glob(os.path.join(source_path, preffix+'*.npz'))
        batched_paths = glob.glob(os.path.join(batched_path, 'batch*.npz'))
        
        sum_source = 0
        sum_s_gesund, sum_s_ill, sum_s_not_certain = 0, 0, 0
        sum_b_gesund, sum_b_ill, sum_b_not_certain = 0, 0, 0
        sum_diff, sum_diff_gesund, sum_diff_ill, sum_diff_not_certain = 0, 0, 0, 0
        
        for s in tqdm(source_paths):
            name = s.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0]
            print(f'We are checking {name}')
            
            s_data = np.load(s)
            s_X, s_y = s_data['X'], s_data['y']
            
            s_gesund = np.flatnonzero(s_y == 0).shape[0]
            s_ill = np.flatnonzero(s_y == 1).shape[0]
            s_not_certain = np.flatnonzero(s_y == 2).shape[0]
            
            sum_source += s_X.shape[0]
            sum_s_ill += s_ill
            sum_s_gesund += s_gesund
            sum_s_not_certain += s_not_certain
            
            if sum_source != sum_s_ill + sum_s_gesund + sum_s_not_certain:
                print(f'Warning!!! source {s} has strange size')
            
            sum_batch = 0
            sum_b_gesund, sum_b_ill, sum_b_not_certain = 0, 0, 0
            
            b_all, b_gesund, b_ill, b_not_certain = 0, 0, 0, 0
            for b in batched_paths:
                b_data = np.load(b)
                b_X, b_y = b_data['X'], b_data['y']
                if b_X.shape[0] != config.BATCH_SIZE:
                    print(f'Warning!!! batch {b} has more then batch_size shape')
                
                sum_batch += b_X.shape[0]
                sum_b_gesund += np.flatnonzero(b_y == 0).shape[0]
                sum_b_ill += np.flatnonzero(b_y == 1).shape[0]
                sum_b_not_certain += np.flatnonzero(b_y == 2).shape[0]
                
                #indx = np.flatnonzero(np.core.defchararray.find(b_data['PatientName'], name) != -1)
                indx = np.flatnonzero(b_data['PatientName'] == name) 
                b_all += indx.shape[0]
                b_gesund += np.flatnonzero(b_y[indx] == 0).shape[0]
                b_ill += np.flatnonzero(b_y[indx] == 1).shape[0]
                b_not_certain += np.flatnonzero(b_y[indx] == 2).shape[0]
                
                if np.flatnonzero(b_y == 2).shape[0] > 0:
                    print(f'Warning!!! batch {b} has not_certain data')
            
            sum_diff += s_X.shape[0] - b_all
            sum_diff_gesund += s_gesund - b_gesund
            sum_diff_ill += s_ill - b_ill
            sum_diff_not_certain += s_not_certain - b_not_certain
            
            print(f'Differences: healthy - {s_gesund - b_gesund}, ill - {s_ill - b_ill}, not_certain - {s_not_certain - b_not_certain}, all - {s_X.shape[0] - b_all}')
                
            print('----------------------------------------')
        
        print('Sources (all, gesund, ill, not_certain):', sum_source, sum_s_gesund, sum_s_ill, sum_s_not_certain)
        print('Batches (all, gesund, ill, not_certain):', sum_batch, sum_b_gesund, sum_b_ill, sum_b_not_certain)
        print('Diffs (all, gesund, ill, not_certain):', sum_diff, sum_diff_gesund, sum_diff_ill, sum_diff_not_certain)        
        if sum_source - sum_batch - sum_diff_gesund - sum_diff_ill - sum_diff_not_certain != 0:
            print(f'Warning!!! batch {b} has not_certain data')
            
            
if __name__ == '__main__':
    checker = Checker()
    
    #checker.check_dat_npz(config.DATA_PATHS, r'/work/users/mi186veva/data_preprocessed/combi')
    #checker.check_source_and_shuffled( r'/work/users/mi186veva/data_bea/ColonData/raw_3d_weights',  r'/work/users/mi186veva/data_bea/ColonData/raw_3d_weights/shuffled')
    checker.check_source_and_batched(r'../data_preprocessed/raw_3d_weighted/shuffled', r'../data_preprocessed/raw_3d_weighted/batch_sized')
        
        
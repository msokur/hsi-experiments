from hypercube_data import *
import numpy as np
import os
import glob
import config
import data_loader
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
        self.dat_source_paths = dat_source_paths
        self.raw_npz_path = npz_path
        
        npz_paths = glob.glob(os.path.join(npz_path, '*.npz'))
        
        dat_paths = []
        for path_dir in dat_source_paths:
            dat_paths += glob.glob(os.path.join(path_dir, '*.dat'))
        
        for npz in npz_paths:           
            name = npz.split("/")[-1].split(".")[0]
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
            
            print('----------------------------------------')
          
    def check_raw_npz_and_aug(self, raw_npz_path, aug_path):
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
        source_paths = glob.glob(os.path.join(source_path, '*.npz'))
        shuffled_paths = glob.glob(os.path.join(shuffled_path, 'shuffle*.npz'))
                
        for s in tqdm(source_paths):
            name = s.split("/")[-1].split(".")[0]
            print(f'We are checking {name}')
            
            s_data = np.load(s)
            s_gesund = s_data['X'][np.where(s_data['y'] == 0)].shape[0]
            s_ill = s_data['X'][np.where(s_data['y'] == 1)].shape[0]
            s_not_certain = s_data['X'][np.where(s_data['y'] == 2)].shape[0]
            
            sh_all, sh_gesund, sh_ill, sh_not_certain = 0, 0, 0, 0
            for shuff in shuffled_paths:
                sh_data = np.load(shuff)
                
                indx = np.flatnonzero(np.core.defchararray.find(sh_data['PatientName'], name) != -1)
                sh_all += indx.shape[0]
                sh_gesund += np.where(sh_data['y'][indx] == 0)[0].shape[0]
                sh_ill += np.where(sh_data['y'][indx] == 1)[0].shape[0]
                sh_not_certain += np.where(sh_data['y'][indx] == 2)[0].shape[0]
                
            print(f'Differences: healthy - {s_gesund - sh_gesund}, ill - {s_ill - sh_ill}, not_certain - {s_not_certain - sh_not_certain}, all - {s_data["X"].shape[0] - sh_all}')
                
            print('----------------------------------------')
            
    def check_source_and_batched(self, source_path, batched_path, preffix=''):
        source_paths = glob.glob(os.path.join(source_path, preffix+'*.npz'))
        batched_paths = glob.glob(os.path.join(batched_path, 'batch*.npz'))
        
        sum_source = 0
        sum_s_gesund, sum_s_ill, sum_s_not_certain = 0, 0, 0
        sum_b_gesund, sum_b_ill, sum_b_not_certain = 0, 0, 0
        sum_diff, sum_diff_gesund, sum_diff_ill, sum_diff_not_certain = 0, 0, 0, 0
        
        for s in tqdm(source_paths):
            name = s.split("/")[-1].split(".")[0]
            print(f'We are checking {name}')
            
            s_data = np.load(s)
            s_X, s_y = s_data['X'], s_data['y']
            
            s_gesund = np.flatnonzero(s_y == 0).shape[0]
            s_ill =np.flatnonzero(s_y == 1).shape[0]
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
                
                indx = np.flatnonzero(np.core.defchararray.find(b_data['PatientName'], name) != -1)
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
    
    checker.check_source_and_shuffled(config.AUGMENTED_PATH, config.SHUFFLED_PATH)
    #checker.check_source_and_batched(config.AUGMENTED_PATH, r'/work/users/mi186veva/data_preprocessed/augmented/batch_sized')
        
        
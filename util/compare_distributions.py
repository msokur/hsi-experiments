import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import config

from statsmodels.stats import weightstats
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import math
from scipy.stats import ks_2samp

class DistributionsChecker:
    # path - from which we will compare archives
    # prints - if print some intermediate information
    def __init__(self, path, prints=False):
        self.path = path  
        self.test_paths = glob(os.path.join(self.path, '*.npz'))[:10]
        self.all_data = np.array(self.get_data(self.test_paths))
        self.prints = prints  
    
    @staticmethod
    #For 3d data: we don't need to compare every patch with other patches
    #It's enouph to compare only centers of patches
    #As a result the comparison is faster
    def get_centers(data):
        center = math.floor(data.shape[1] / 2)
        return data[:, center, center, ...]
    
    @staticmethod
    def get_data(paths, feature_index=-1):
        all_data = []

        for p in tqdm(paths):
            data = np.load(p)
            data = DistributionsChecker.get_centers(data['X'])

            all_data += list(data)
            if feature_index >= 0:
                all_data += list(data[:, feature_index])

        return all_data
    
    def z_test(self, d1, d2):
        #check std
        std1 = np.std(d1)
        std2 = np.std(d2)
        if self.prints:
            print(f'std1 - {std1}, std2 - {std2}')
        
        if np.abs(std1 - std2) < config.Z_TEST_STD_DELTA:
            z, p_value = weightstats.ztest(d1, x2=d2, value=0)
            if self.prints:
                print('z-score and p_value:', z, p_value)
            if p_value > config.Z_TEST_P_VALUE:
                return True
        else:
            print('WARNING! Not possible to provide Z-test: distributions(std) are too different')
        return False
    
    # Method for comparing of two derived distributions
    # d1, d2 - two arrays, which distributions we want to compare
    # Wikipedia: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    # Good explanation: https://www.youtube.com/watch?v=ZO2RmSkXK3c
    def kolmogorov_smirnov_test(self, d1, d2):
        ks = ks_2samp(d1, d2)
        if self.prints:
            print(ks)
        if ks.pvalue > config.KS_TEST_P_VALUE:
            return True
        return False


    def compare_shuffled_distributions(self, test_archive_index=0):
        test_archive = np.load(self.test_paths[test_archive_index])
        test_data = self.get_centers(test_archive['X'])
        
        results = []
        for i in range(config.LAST_NM - config.FIRST_NM):
            z_result = True
            if config.Z_TEST:
                z_result = self.z_test(test_data[:, i], self.all_data[:, i])

            ks_result = self.kolmogorov_smirnov_test(self.all_data[:, i], test_data[:, i])
            results.append(z_result & ks_result)
        if np.all(results):
            return True
        return False
    
    def get_small_database_for_tuning(self):
        result = False
        for i in range(len(self.test_paths)):
            result = self.compare_shuffled_distributions(test_archive_index=i)
            if result:
                break
        if not result:
            raise ValueError(f'Error! No appropriate small archive for {self.path} were found (try to change requirements - config.KS_TEST_P_VALUE)')
        print(f'Choosen index for small database: {i}')
        return i
    
    def test_all_archives_in_folder(self):
        for i in tqdm(range(len(self.test_paths))):
            result = self.compare_shuffled_distributions(test_archive_index=i)
            print(f'Archive {i} can be used: {result}')

if __name__ == '__main__':
    dc = DistributionsChecker('/work/users/mi186veva/data_3d/raw_3d_svn/shuffled', prints=False)
    print(dc.test_all_archives_in_folder())        
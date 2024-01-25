import sys
import os
import inspect
from typing import List

import numpy as np
from tqdm import tqdm
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_utils.dataset import Dataset
from configuration.keys import DistroCheckKeys as DCK

"""
This class is about choosing a representative small dataset for speed up the training
Detailed description: 
https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/How-to-choose-a-representative-smaller-dataset-for-faster-training-(comparison-of-distributions-of-two-datasets)

There are several functionality parts:
1. Comparing distributions:
- z_test
- kolmogorov_smirnov_test
2. Reading data:
- get_centers
- get_data
3. Choosing a representative archive from all archives from the given folder:
- compare_shuffled_distributions
- get_small_database_for_tuning
4. Checking all archives in the given folder: test_all_archives_in_folder
"""


class DistributionsChecker:
    def __init__(self, paths: List[str], dataset: Dataset, config_distribution: dict):
        """
        Args:
            paths: paths with the archives to compare
            config_distribution: configuration parameters
        """
        self.test_paths = paths
        self.path = os.path.split(paths[0])[0]
        self.dataset = dataset
        self.CONFIG_DISTRIBUTION = config_distribution
        self.all_data = np.array(self.get_all_data(paths=self.test_paths))
        self.prints = self.CONFIG_DISTRIBUTION[DCK.PRINTS]

    @staticmethod
    def get_centers(data: np.ndarray) -> np.ndarray:
        """ Get the center of a 3d patch.

        For 3d data: we don't need to compare every patch with the other patches
        It's enough to compare only centers of patches.
        As a result the comparison is faster.
        Args:
            data - 3d data with shapes (number_of_samples, patch_size, patch_size, features),
            for example (50000, 5, 5, 92)
        Returns:
            centers of patches with shapes (number_of_samples, features)
            for the example above (50000, 92)
        """
        center = math.floor(data.shape[1] / 2)
        return data[:, center, center, ...]

    def get_all_data(self, paths, feature_index=-1):
        """
        Read data from the given path.
        It's also possible to get data with the given feature_index (for example, get only 4th feature)

        Returns:
            concatenated data with shapes (number_of_samples, features). For 3d data only the centered patches will be
            used.
        """
        all_data = []

        for p in tqdm(paths):
            data = self.get_data(path=p, feature_index=feature_index)
            all_data += list(data)

        return all_data

    def get_data(self, path, feature_index=-1) -> np.ndarray:
        """
        Read data from the given path.
        It's also possible to get data with the given feature_index (for example, get only 4th feature)

        Returns:
            data with shapes (number_of_samples, feature(s)). For 3d data only the centered patches will be used.
        """
        shape = self.dataset.get_meta_shape(paths=[path])
        data = self.dataset.get_X(path=path, shape=shape)
        if shape.__len__() > 2:
            data_1d = self.get_centers(data=data)
        else:
            data_1d = data

        if feature_index >= 0:
            return data_1d[..., feature_index]
        else:
            return data_1d[...]

    def z_test(self, d1, d2):
        """ z_test for comparing of 2 distributions with the same std: d1 and d2

        Returns:
            True if distributions are the same, False if not the same.
            Or it prints a warning if stds of distributions are not the same
        Wikipedia: https://en.wikipedia.org/wiki/Z-test
        A very good explanation (german): https://statistikgrundlagen.de/ebook/chapter/z-test-gausstest/
        """
        # check std
        std1 = np.std(d1)
        std2 = np.std(d2)
        if self.prints:
            print(f'std1 - {std1}, std2 - {std2}')

        if np.abs(std1 - std2) < self.CONFIG_DISTRIBUTION[DCK.Z_TEST_STD_DELTA]:
            from statsmodels.stats import weightstats
            z, p_value = weightstats.ztest(d1, x2=d2, value=0)
            if self.prints:
                print('z-score and p_value:', z, p_value)
            if p_value > self.CONFIG_DISTRIBUTION[DCK.Z_TEST_P_VALUE]:
                return True
        else:
            print('WARNING! Distributions (std) are too different, Z-test results '
                  'would be meaningless. Use kolmogorov_smirnov_test().')
        return False

    def kolmogorov_smirnov_test(self, d1, d2):
        from scipy.stats import ks_2samp
        """ Method for comparing of two derived distributions.

        Args:
            d1, d2 - two arrays, whose distributions we want to compare.
        Returns:
            True if distributions are the same, False if not the same
        Wikipedia: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        Good explanation: https://www.youtube.com/watch?v=ZO2RmSkXK3c
        """
        ks = ks_2samp(d1, d2)
        if self.prints:
            print(ks)
        if ks.pvalue > self.CONFIG_DISTRIBUTION[DCK.KS_TEST_P_VALUE]:
            return True
        return False

    def compare_shuffled_distributions(self, test_archive_index=0):
        """ Compare distributions separately for each feature.
        Args:
            test_archive_index: index of archive to compare to all the other archives
        Returns:
            True if archive with test_archive_index has the same distribution, False if not the same
        """
        test_data = self.get_data(path=self.test_paths[test_archive_index])

        results = []
        for i in range(test_data.shape[-1]):
            z_result = True
            if self.CONFIG_DISTRIBUTION[DCK.Z_TEST]:
                z_result = self.z_test(test_data[:, i], self.all_data[:, i])

            ks_result = self.kolmogorov_smirnov_test(self.all_data[:, i], test_data[:, i])
            results.append(z_result & ks_result)
        if np.all(results):
            return True
        return False

    def get_small_database_for_tuning(self):
        """ Get an archive with the same distribution as the whole database.
        Returns:
            An index of the first appropriate archive from path with the same distribution as the whole database
            Or raises an error if no appropriate archives were found
        """
        idx = 0
        result = False
        print("--- Test dataset ---")
        for i in tqdm(range(len(self.test_paths))):
            result = self.compare_shuffled_distributions(test_archive_index=i)
            if result:
                idx = i
                break
        if not result:
            raise ValueError(
                f'Error! No matching small archive for {self.path} '
                f'found. Try changing the requirements (KS_TEST_P_VALUE) in yor DistributionsCheck file.')
        print(f'Index of the chosen small database: {idx}')
        return idx

    def test_all_archives_in_folder(self):
        """
        Prints for each archive of the folder if the distribution of archive is the same with the whole dataset (True)
        And False if not the same
        """
        for i in tqdm(range(len(self.test_paths))):
            result = self.compare_shuffled_distributions(test_archive_index=i)
            print(f'Archive {i} can be used: {result}')


if __name__ == '__main__':
    dc = DistributionsChecker('/work/users/mi186veva/data_3d/raw_3d_svn/shuffled')
    print(dc.test_all_archives_in_folder())

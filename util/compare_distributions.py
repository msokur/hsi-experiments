import sys
import os
import inspect
from typing import List, Union

import numpy as np
from tqdm import tqdm
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_utils.dataset import Dataset
from data_utils.data_storage import DataStorage
from configuration.keys import DistroCheckKeys as DCK
from configuration.parameter import FEATURE_X

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
    def __init__(self, local_config: dict, paths: List[str], dataset: Dataset, base_paths: List[str] = None,
                 base_data_storage: Union[Dataset, DataStorage] = None):
        """
        Args:
            local_config: configuration parameters
            paths: paths with the archives to compare
        """
        self.local_config = local_config
        self.test_paths = paths
        self.path = os.path.split(paths[0])[0]
        self.dataset = dataset
        if base_paths is None:
            self.all_data = self.get_all_data(paths=self.test_paths, storage=dataset)
        else:
            if base_data_storage is None:
                raise ValueError("Base Data Storage is None, please insert a 'Dataset' or 'DataStorge' object when"
                                 "base_paths is not None!")
            self.all_data = self.get_all_data(paths=base_paths, storage=base_data_storage)
        self.prints = local_config[DCK.PRINTS]

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
        return data[:, center, center, ...].copy()

    def get_all_data(self, paths: List[str], storage: Union[DataStorage, Dataset], feature_index=-1):
        """
        Read data from the given path.
        It's also possible to get data with the given feature_index (for example, get only 4th feature)

        Returns:
            concatenated data with shapes (number_of_samples, features). For 3d data only the centered patches will be
            used.
        """
        all_data = []

        for p in tqdm(paths):
            if isinstance(storage, Dataset):
                data = self.get_data(data=storage.get_X(path=p), feature_index=feature_index)
            elif isinstance(storage, DataStorage):
                data = self.get_data(data=storage.get_data(data_path=p, data_name=FEATURE_X),
                                     feature_index=feature_index)
            else:
                raise ValueError("Storage type for all data reading wrong!")
            all_data.append(data)

        return np.concatenate(all_data, axis=0)

    def get_data(self, data: np.ndarray, feature_index=-1) -> np.ndarray:
        """
        Read data from the given path.
        It's also possible to get data with the given feature_index (for example, get only 4th feature)

        Returns:
            data with shapes (number_of_samples, feature(s)). For 3d data only the centered patches will be used.
        """
        if data.shape.__len__() > 2:
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

        if np.abs(std1 - std2) < self.local_config[DCK.Z_TEST_STD_DELTA]:
            from statsmodels.stats import weightstats
            z, p_value = weightstats.ztest(x1=d1, x2=d2, value=0)
            if self.prints:
                print('z-score and p_value:', z, p_value)
            return p_value > self.local_config[DCK.Z_TEST_P_VALUE]

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

        return ks.pvalue > self.local_config[DCK.KS_TEST_P_VALUE]

    def compare_shuffled_distributions(self, test_archive_index=0):
        """ Compare distributions separately for each feature.
        Args:
            test_archive_index: index of archive to compare to all the other archives
        Returns:
            True if archive with test_archive_index has the same distribution, False if not the same
        """
        test_data = self.get_data(data=self.dataset.get_X(path=self.test_paths[test_archive_index]))
        results = []
        for i in range(test_data.shape[-1]):
            z_result = True
            if self.local_config[DCK.Z_TEST]:
                z_result = self.z_test(test_data[:, i], self.all_data[:, i])

            ks_result = self.kolmogorov_smirnov_test(self.all_data[:, i], test_data[:, i])
            results.append(z_result & ks_result)

        return np.all(results)

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
        Returns an array for each archive of the folder if the distribution of archive is the same with the whole
        dataset (True) And False if not the same
        """
        usable = []
        for i in tqdm(range(len(self.test_paths))):
            result = self.compare_shuffled_distributions(test_archive_index=i)
            if self.prints:
                print(f"Archive '{self.test_paths[i]}' can be used: {result}")
            usable.append(result)

        return usable


if __name__ == '__main__':
    from configuration import get_config as config
    from provider import get_dataset, get_data_storage
    from glob import glob
    from data_utils.data_storage import DataStorageNPZ

    dataset_ = get_dataset(typ="tfr", config=config, data_storage=get_data_storage(typ="npz"))
    dc_config = {
        "PRINTS": False,
        "Z_TEST": True,
        "Z_TEST_P_VALUE": 0.05,
        "Z_TEST_STD_DELTA": 0.01,
        "KS_TEST_P_VALUE": 0.05
    }
    main_path = r"C:\Users\benny\Desktop\Arbeit\data\data_3d\hno\3x3\svn_T\no_smooth"
    test_paths_ = glob(os.path.join(main_path, "shuffled_tfr_test_2", "*.tfrecord"))
    main_paths_ = glob(os.path.join(main_path, "patient_data_npz", "*.npz"))
    dc = DistributionsChecker(dc_config, paths=[test_paths_[0]], dataset=dataset_,
                              base_paths=main_paths_, base_data_storage=DataStorageNPZ())
    print(dc.test_all_archives_in_folder())

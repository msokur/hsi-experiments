from typing import List

import numpy as np
import os

from data_utils.shuffle import Shuffle
from data_utils.dataset import Dataset, get_meta_files
from util.compare_distributions import DistributionsChecker

from configuration.keys import PathKeys as PK


class SmallShuffle:
    def __init__(self, config, shuffle_class: Shuffle, dataset: Dataset, set_seed: bool = True):
        self.config = config
        self.shuffle_class = shuffle_class
        self.data_storage = shuffle_class.data_storage
        self.raw_path = self.shuffle_class.raw_path
        self.dataset = dataset
        self.set_seed = set_seed
        print("----- Load DistributionChecker for smaller shuffle dataset -----")
        self.distro_check = DistributionsChecker(local_config=self.config.CONFIG_DISTRIBUTION, paths=[""],
                                                 dataset=self.dataset,
                                                 base_paths=self.data_storage.get_paths(storage_path=self.raw_path),
                                                 base_data_storage=self.data_storage)

    def shuffle(self, use_piles: List[int] = None, piles_count: int = 1):
        if self.set_seed:
            np.random.seed(seed=42)

        self.shuffle_class.check_piles_number()
        piles_number = self.shuffle_class.piles_number
        usable_piles = [i for i in range(piles_number)]
        if use_piles is None:
            use_piles = self.__get_use_piles(usable_piles=usable_piles, size=piles_count)
        print("----- Create small shuffle set started -----")
        sh_paths = self.shuffle_and_check(use_piles=use_piles, usable_piles=usable_piles)
        print(f"Representative shuffle files are: {', '.join(os.path.basename(p) for p in sh_paths)}")
        print("----- Create small shuffle set finished -----")

    def shuffle_and_check(self, use_piles: List[int], usable_piles: List[int]) -> List[str]:
        self.shuffle_class.shuffle(use_piles=use_piles)
        test_paths = self.__get_test_paths(use_piles=use_piles)

        print("----- Check Distribution started -----")
        self.distro_check.test_paths = test_paths
        use_paths = self.distro_check.test_all_archives_in_folder()
        print("----- Check Distribution finished -----")

        if np.all(use_paths):
            return test_paths
        else:
            usable_piles_new = list(set(usable_piles) - set(use_piles))
            use_piles_new = self.__get_use_piles(usable_piles=usable_piles_new,
                                                 size=np.count_nonzero(~np.array(use_paths)))
            true_test_paths = [p for p, t in zip(test_paths, use_paths) if t]
            delete_test_paths = [p for p, t in zip(test_paths, use_paths) if not t]
            print(f"Shuffle files not representative: {', '.join(os.path.basename(p) for p in delete_test_paths)}")
            print("--- Start new shuffle and check ---")
            self.__delete_paths(delete_paths=delete_test_paths)
            return true_test_paths + self.shuffle_and_check(use_piles=use_piles_new, usable_piles=usable_piles_new)

    @staticmethod
    def __get_use_piles(usable_piles: List[int], size: int) -> List[int]:
        return np.random.choice(a=usable_piles, size=size, replace=False)

    def __delete_paths(self, delete_paths: List[str]):
        meta_paths = get_meta_files(paths=delete_paths, typ=self.shuffle_class.dataset_typ)
        for p, m_p in zip(delete_paths, meta_paths):
            os.remove(path=p)
            os.remove(path=m_p)

    def __get_test_paths(self, use_piles: List[int]) -> List[str]:
        test_paths_ = self.dataset.get_dataset_paths(root_paths=self.config.CONFIG_PATHS[PK.SHUFFLED_PATH])
        test_paths = []
        for t_p in test_paths_:
            num = int(os.path.basename(os.path.splitext(t_p)[0]).split("_")[-1])
            if num in use_piles:
                test_paths.append(t_p)

        return test_paths

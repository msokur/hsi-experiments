import os
from glob import glob
from typing import List, Dict, Union, Iterable

import numpy as np

from data_utils.data_archive import DataArchive


class DataArchiveNPZ(DataArchive):
    @staticmethod
    def get_path(file: np.lib.npyio.NpzFile) -> str:
        return file.fid.name

    @staticmethod
    def get_paths(archive_path: str) -> List[str]:
        return glob(os.path.join(archive_path, "*.npz"))

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1].split(".npz")[0]

    def all_data_generator(self, archive_path: str) -> Iterable:
        paths = self.get_paths(archive_path=archive_path)
        for path in paths:
            yield np.load(file=path)

    @staticmethod
    def get_datas(data_path: str) -> Union[np.ndarray, Iterable, int, float, tuple, dict]:
        return np.load(file=data_path)

    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        data = self.get_datas(data_path=data_path)
        return data[data_name]

    def get_batch_data(self, batch_path: str, name: str):
        return self.get_data(data_path=batch_path, data_name=name)

    def get_batch_datas(self, batch_path: str, X: str, y: str, weights: str = None):
        data = self.get_datas(data_path=batch_path)
        if weights is not None:
            return data[X], data[y], data[weights]
        else:
            return data[X], data[y]

    def save_batch_datas(self, save_path: str, data: Dict[str, Union[np.ndarray, list]], data_indexes: np.ndarray,
                         batch_file_name: str, split_size: int, save_dict_names: List[str]) -> Dict[str, np.ndarray]:
        # ---------------splitting into archives----------
        chunks = data_indexes.shape[0] // split_size
        chunks_max = chunks * split_size

        if chunks > 0:
            data_ = {k: np.array_split(a[data_indexes][:chunks_max], chunks) for k, a in data.items()}

            idx = len(glob(os.path.join(save_path, "*.npz")))
            for row in range(chunks):
                arch = {}
                for i, n in enumerate(save_dict_names):
                    arch[n] = data_[n][row]

                np.savez(os.path.join(save_path, f"{batch_file_name}{idx}"), **arch)
                idx += 1

        # ---------------saving of the non equal last part for the future partition---------
        rest = {k: a[data_indexes][chunks_max:] for k, a in data.items()}
        # ---------------saving of the non equal last part for the future partition---------
        return rest

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        np.savez(file=os.path.join(save_path, group_name), **datas)

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        npz_data = np.load(file=save_path)
        datas = {k: v for k, v in npz_data.items()}
        datas[data_name] = data
        main_path, name = os.path.split(p=save_path)
        self.save_group(save_path=main_path, group_name=name, datas=datas)

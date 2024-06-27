import os
from glob import glob
from typing import List, Dict, Union, Iterable

import numpy as np

from data_utils.data_storage import DataStorage


class DataStorageNPZ(DataStorage):
    @staticmethod
    def get_path(file) -> str:
        return file.fid.name

    @staticmethod
    def get_paths(storage_path: str) -> List[str]:
        return sorted(glob(os.path.join(storage_path, "*.npz")))

    def all_data_generator(self, storage_path: str) -> Iterable:
        paths = self.get_paths(storage_path=storage_path)
        for path in paths:
            yield np.load(file=path)

    @staticmethod
    def get_datas(data_path: str) -> Union[np.ndarray, Iterable, int, float, tuple, dict]:
        if not data_path.endswith(".npz"):
            data_path += ".npz"
        return np.load(file=data_path)

    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        data = self.get_datas(data_path=data_path)
        return data[data_name]

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        if not os.path.exists(save_path):
            os.mkdir(path=save_path)
        np.savez(file=os.path.join(save_path, group_name), **datas)

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        npz_data = np.load(file=save_path, allow_pickle=True)
        datas = {k: v for k, v in npz_data.items()}
        datas[data_name] = data
        main_path, name = os.path.split(p=save_path)
        self.save_group(save_path=main_path, group_name=name, datas=datas)

    def append_data(self, file_path: str, append_datas: Dict[str, np.ndarray]) -> None:
        if not file_path.endswith(".npz"):
            file_path += ".npz"

        if not os.path.exists(path=file_path):
            super()._file_not_found(file=file_path)

        data = np.load(file=file_path)
        super()._check_keys(name=os.path.split(file_path)[-1], base_keys=set(data.keys()),
                            append_keys=set(append_datas.keys()))
        new_data = {}
        for k, v in data.items():
            new_data[k] = np.concatenate((v, append_datas[k]), axis=0)
        np.savez(file=file_path, **new_data)

    @staticmethod
    def get_extension() -> str:
        return ".npz"

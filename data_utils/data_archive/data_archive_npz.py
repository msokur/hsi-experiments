import os
from glob import glob
from typing import List, Dict, Union, Iterable

import numpy as np

from data_utils.data_archive import DataArchive


class DataArchiveNPZ(DataArchive):
    @staticmethod
    def get_path(file) -> str:
        return file.fid.name

    @staticmethod
    def get_paths(archive_path: str) -> List[str]:
        return sorted(glob(os.path.join(archive_path, "*.npz")))

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1].split(".npz")[0]

    def all_data_generator(self, archive_path: str) -> Iterable:
        paths = self.get_paths(archive_path=archive_path)
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
        npz_data = np.load(file=save_path)
        datas = {k: v for k, v in npz_data.items()}
        datas[data_name] = data
        main_path, name = os.path.split(p=save_path)
        self.save_group(save_path=main_path, group_name=name, datas=datas)

import abc
from typing import List, Dict, Union

import numpy as np
import os

import zarr


class DataArchive:
    @staticmethod
    @abc.abstractmethod
    def get_path(file) -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_paths(archive_path: str) -> List[str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name(path: str) -> str:
        pass

    @abc.abstractmethod
    def all_data_generator(self, archive_path: str):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_datas(data_path: str):
        pass

    @abc.abstractmethod
    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save_batch_arrays(self, save_path: str, data: Union[zarr.Group, Dict[str, Union[np.ndarray, list]]],
                          data_indexes: np.ndarray, batch_file_name: str, split_size: int,
                          save_dict_names: List[str]) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        pass

    @abc.abstractmethod
    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def delete_group(delete_path):
        pass

    def delete_archive(self, delete_path):
        datas_to_delete = self.get_paths(archive_path=delete_path)
        for delete in datas_to_delete:
            self.delete_group(delete_path=delete)
        os.remove(delete_path)

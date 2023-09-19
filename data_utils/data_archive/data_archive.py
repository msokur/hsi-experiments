import abc
from shutil import rmtree
from typing import List, Dict, Iterable

import numpy as np


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
    def all_data_generator(self, archive_path: str) -> Iterable:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_datas(data_path: str):
        pass

    @abc.abstractmethod
    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        pass

    @abc.abstractmethod
    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        pass

    @staticmethod
    def delete_archive(delete_path):
        rmtree(delete_path)

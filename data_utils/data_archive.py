import abc
from typing import List, Dict

import numpy as np
import zarr
import os
from glob import glob

from configuration.parameter import (
    PAT_CHUNKS, D3_PAT_CHUNKS,
)


class DataArchive:
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

    @staticmethod
    @abc.abstractmethod
    def get_data(data_path: str, data_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        pass

    @abc.abstractmethod
    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        pass


class DataArchiveNPZ(DataArchive):
    @staticmethod
    def get_paths(archive_path: str) -> List[str]:
        return glob(os.path.join(archive_path, "*.npz"))

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1].split(".npz")[0]

    def all_data_generator(self, archive_path: str):
        paths = self.get_paths(archive_path=archive_path)
        for path in paths:
            yield np.load(file=path)

    @staticmethod
    def get_datas(data_path: str):
        return np.load(file=data_path)

    @staticmethod
    def get_data(data_path: str, data_name: str) -> np.ndarray:
        data = np.load(file=data_path)
        return data[data_name]

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        np.savez(file=os.path.join(save_path, group_name), **datas)

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        npz_data = np.load(file=save_path)
        datas = {k: v for k, v in npz_data.items()}
        datas[data_name] = data
        main_path, name = os.path.split(p=save_path)
        self.save_group(save_path=main_path, group_name=name, datas=datas)


class DataArchiveZARR(DataArchive):
    @staticmethod
    def get_paths(archive_path: str) -> List[str]:
        paths = []
        data = zarr.open_group(store=archive_path)
        for group in data.group_keys():
            paths.append(os.path.abspath(archive_path) + f"/{data[group].path}")

        return paths

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1]

    def all_data_generator(self, archive_path: str):
        paths = self.get_paths(archive_path=archive_path)
        for path in paths:
            yield zarr.open_group(store=path, mode="r")

    @staticmethod
    def get_datas(data_path: str):
        return zarr.open_group(store=data_path, mode="r")

    @staticmethod
    def get_data(data_path: str, data_name: str) -> np.ndarray:
        data = zarr.open_group(store=data_path, mode="r")
        return data[data_name]

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        pat = root.create_group(name=group_name, overwrite=True)
        for k, v in datas.items():
            pat.array(name=k, data=v, chunks=self.__get_chunks__(len(v.shape)))

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        root.array(name=data_name, data=data, chunks=self.__get_chunks__(len(data.shape)), overwrite=True)

    @staticmethod
    def __get_chunks__(data_length: int) -> tuple:
        if data_length < 3:
            return PAT_CHUNKS
        else:
            return D3_PAT_CHUNKS

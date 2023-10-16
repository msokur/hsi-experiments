import os
from typing import List, Dict, Iterable

import numpy as np
import zarr

from configuration.parameter import (
    BLOCK_SIZE, D3_CHUNK
)
from data_utils.data_archive import DataArchive


class DataArchiveZARR(DataArchive):
    @staticmethod
    def get_path(file: zarr.Group) -> str:
        return file.attrs.store.path

    @staticmethod
    def get_paths(archive_path: str) -> List[str]:
        paths = []
        data = zarr.open_group(store=archive_path)
        for group in data.group_keys():
            paths.append(os.path.join(os.path.abspath(archive_path), data[group].path))

        return paths

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1]

    def all_data_generator(self, archive_path: str) -> Iterable:
        paths = self.get_paths(archive_path=archive_path)
        for path in paths:
            yield zarr.open_group(store=path, mode="r")

    @staticmethod
    def get_datas(data_path: str) -> zarr.Group:
        return zarr.open_group(store=data_path, mode="r")

    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        data = self.get_datas(data_path=data_path)
        return data[data_name]

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        pat = root.create_group(name=group_name, overwrite=True)
        for k, v in datas.items():
            pat.array(name=k, data=v, chunks=self.__get_chunks__(v.shape))

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        root = zarr.open_group(store=save_path, mode="r+")
        root.array(name=data_name, data=data, chunks=self.__get_chunks__(data.shape), overwrite=True)

    @staticmethod
    def __get_chunks__(data_shape: tuple) -> tuple:
        block_count = -(-data_shape[0] // BLOCK_SIZE)
        chunk = -(-data_shape[0] // block_count)
        if len(data_shape) < 3:
            return (chunk,)
        else:
            return (chunk,) + D3_CHUNK

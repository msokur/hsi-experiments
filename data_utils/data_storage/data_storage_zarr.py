import os
from typing import List, Dict, Iterable

import numpy as np
import zarr

from configuration.parameter import (
    BLOCK_SIZE, D3_CHUNK
)
from data_utils.data_storage import DataStorage


class DataStorageZARR(DataStorage):
    @staticmethod
    def get_path(file: zarr.Group) -> str:
        return file.attrs.store.path

    @staticmethod
    def get_paths(storage_path: str) -> List[str]:
        paths = []
        data = zarr.open_group(store=storage_path)
        root_path = os.path.abspath(storage_path)
        for group in data.group_keys():
            paths.append(os.path.join(root_path, data[group].path))

        return sorted(paths)

    def all_data_generator(self, storage_path: str) -> Iterable:
        paths = self.get_paths(storage_path=storage_path)
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
            pat.array(name=k, data=v, chunks=self.__get_chunks(v.shape))

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        root = zarr.open_group(store=save_path, mode="r+")
        root.array(name=data_name, data=data, chunks=self.__get_chunks(data.shape), overwrite=True)

    def append_data(self, file_path: str, append_datas: Dict[str, np.ndarray]) -> None:
        if not os.path.exists(path=file_path):
            super()._file_not_found(file=file_path)

        data = zarr.open_group(store=file_path, mode="r+")
        super()._check_keys(name=os.path.split(file_path)[-1], base_keys=set(data.keys()),
                            append_keys=set(append_datas.keys()))
        for k, v in data.items():
            v.append(append_datas[k], axis=0)

    @staticmethod
    def __get_chunks(data_shape: tuple) -> tuple:
        block_count = -(-data_shape[0] // BLOCK_SIZE)
        chunk = -(-data_shape[0] // block_count)
        if len(data_shape) < 3:
            return (chunk,)
        else:
            return (chunk,) + D3_CHUNK

    @staticmethod
    def get_extension() -> str:
        return ".zarr"

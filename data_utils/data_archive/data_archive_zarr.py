import os
from shutil import rmtree
from typing import List, Dict, Union, Iterable

import numpy as np
import zarr

from configuration.parameter import (
    PAT_CHUNKS, D3_PAT_CHUNKS, BATCH_IDX, BATCH_ORG_PATH,
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
            paths.append(os.path.abspath(archive_path) + f"/{data[group].path}")

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

    def get_batch_data(self, batch_path: str, X: str, y: str, weights: str = None):
        batch_data = self.get_datas(data_path=batch_path)
        batch_mask = batch_data.get(BATCH_IDX)
        data_path = batch_data.get(BATCH_ORG_PATH)[0]
        data = self.get_datas(data_path=data_path)
        if weights is not None:
            return data[X][...][batch_mask], data[y][...][batch_mask], data[weights][...][batch_mask]
        else:
            return data[X][...][batch_mask], data[y][...][batch_mask]

    def save_batch_datas(self, save_path: str, data: Union[zarr.Group, Dict[str, Union[np.ndarray, list]]],
                         data_indexes: np.ndarray, batch_file_name: str, split_size: int,
                         save_dict_names: List[str]) -> Dict[str, np.ndarray]:
        save_arch = zarr.open_group(store=save_path, mode="a")
        idx = len([k for k in save_arch.group_keys()])
        batch = save_arch.create_group(name=f"{batch_file_name}{idx}", overwrite=True)
        batch.array(name=BATCH_IDX, data=data_indexes, chunks=(split_size,))
        batch.array(name=BATCH_ORG_PATH, data=[self.get_path(file=data)])
        return {k: np.array([]) for k in save_dict_names}

    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        pat = root.create_group(name=group_name, overwrite=True)
        for k, v in datas.items():
            pat.array(name=k, data=v, chunks=self.__get_chunks__(len(v.shape)))

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        root.array(name=data_name, data=data, chunks=self.__get_chunks__(len(data.shape)), overwrite=True)

    @staticmethod
    def delete_group(delete_path):
        rmtree(delete_path)

    @staticmethod
    def __get_chunks__(data_length: int) -> tuple:
        if data_length < 3:
            return PAT_CHUNKS
        else:
            return D3_PAT_CHUNKS

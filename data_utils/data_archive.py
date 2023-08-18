import abc
from typing import List, Dict

import numpy as np
import zarr
import os
from glob import glob


class DataArchive:
    def __init__(self, archive_path: str):
        self.archive_path = archive_path

    @abc.abstractmethod
    def get_paths(self, archive_path: str) -> List[str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name(path: str) -> str:
        pass

    @abc.abstractmethod
    def all_data_generator(self):
        pass

    @abc.abstractmethod
    def get_datas(self, data_path: str):
        pass

    @abc.abstractmethod
    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save_group(self, save_path: str, archive_name: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        pass

    @abc.abstractmethod
    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        pass


class DataArchiveNPZ(DataArchive):
    def get_paths(self, archive_path: str) -> List[str]:
        return glob(os.path.join(archive_path, "*.npz"))

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1].split(".npz")[0]

    def all_data_generator(self):
        paths = self.get_paths(archive_path=self.archive_path)
        for path in paths:
            yield np.load(file=path)

    def get_datas(self, data_path: str):
        return np.load(file=data_path)

    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        data = np.load(file=data_path)
        return data[data_name]

    def save_group(self, save_path: str, archive_name: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        np.savez(file=os.path.join(save_path, group_name), **datas)

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        npz_data = np.load(file=save_path)
        datas = {k: v for k, v in npz_data.items()}
        datas[data_name] = data
        main_path, name = os.path.split(p=save_path)
        self.save_group(save_path=main_path, archive_name="", group_name=name, datas=datas)


class DataArchiveZARR(DataArchive):
    def __init__(self, archive_path: str, archive_name: str, chunks: tuple):
        super().__init__(archive_path)
        self.archive_name = archive_name
        self.chunks = chunks

    def get_paths(self, archive_path: str) -> List[str]:
        paths = []
        zarr_path = os.path.join(archive_path, self.archive_name)
        data = zarr.open_group(store=zarr_path)
        for group in data.group_keys():
            paths.append(os.path.abspath(zarr_path) + f"/{data[group].path}")

        return paths

    @staticmethod
    def get_name(path: str) -> str:
        return os.path.split(path)[-1]

    def all_data_generator(self):
        paths = self.get_paths(archive_path=self.archive_path)
        for path in paths:
            yield zarr.open_group(store=path, mode="r")

    def get_datas(self, data_path: str):
        return zarr.open_group(store=data_path, mode="r")

    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        data = zarr.open_group(store=data_path, mode="r")
        return data[data_name]

    def save_group(self, save_path: str, archive_name: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        root = zarr.open_group(store=os.path.join(save_path, archive_name), mode="a")
        pat = root.create_group(name=group_name, overwrite=True)
        for k, v in datas.items():
            pat.array(name=k, data=v, chunks=self.chunks)

    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        root = zarr.open_group(store=save_path, mode="a")
        root.array(name=data_name, data=data, chunks=self.chunks, overwrite=True)

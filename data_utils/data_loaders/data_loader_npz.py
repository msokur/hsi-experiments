import abc
from typing import List

from data_utils.data_loaders.data_loader import DataLoader
from configuration.keys import PathKeys as PK, DataLoaderKeys as DLK
from glob import glob
import os
import numpy as np


class DataLoaderNPZ(DataLoader):
    def __init__(self, config_dataloader: dict, config_paths: dict, dict_names=None):
        super().__init__(config_dataloader=config_dataloader, config_paths=config_paths, dict_names=dict_names)

    def get_name(self, path: str, delimiter=None) -> str:
        if delimiter is None:
            delimiter = self.CONFIG_PATHS[PK.SYS_DELIMITER]
        return path.split(delimiter)[-1].split(".")[0].split(self.CONFIG_DATALOADER[DLK.NAME_SPLIT])[0]

    @staticmethod
    def get_paths(root_path) -> List[str]:
        return glob(os.path.join(root_path, "*.npz"))

    @abc.abstractmethod
    def file_read(self, path):
        super().file_read(path=path)

    def labeled_spectrum_get_from_archive(self, path: str) -> dict:
        data = np.load(path)
        X, y = data[self.dict_names[0]], data[self.dict_names[1]]

        return super().labeled_spectrum_get_from_X_y(X, y)

    def X_y_dict_save_to_archive(self, destination_path: str, values: dict, name: str) -> None:
        np.savez(os.path.join(destination_path, name), **{n: a for n, a in values.items()})

from typing import List, Dict
import os

import numpy as np
import zarr

from data_utils.data_loaders.data_loader import DataLoader
from configuration.parameter import (
    ZARR_PAT_ARCHIVE,
    PAT_CHUNKS, D3_PAT_CHUNKS,
)


class DataLoaderZARR(DataLoader):
    def __init__(self, config_dataloader: dict, config_paths: dict, dict_names=None):
        super().__init__(config_dataloader=config_dataloader, config_paths=config_paths, dict_names=dict_names)

    def get_name(self, path: str, delimiter=None) -> str:
        return os.path.split(path)[-1]

    @staticmethod
    def get_paths(root_path) -> List[str]:
        paths = []
        zarr_path = os.path.join(root_path, ZARR_PAT_ARCHIVE)
        data = zarr.open_group(store=zarr_path)
        for group in data.group_keys():
            paths.append(os.path.abspath(zarr_path) + f"/{data[group].path}")

        return paths

    def file_read(self, path):
        super().file_read(path=path)

    def labeled_spectrum_get_from_archive(self, path: str) -> dict:
        data = zarr.open_group(store=path, mode="r")
        X, y = data[self.dict_names[0]], data[self.dict_names[1]]

        return super().labeled_spectrum_get_from_X_y(X=X, y=y)

    def X_y_dict_save_to_archive(self, destination_path: str, values: Dict[str, np.ndarray], name: str):
        root = zarr.open_group(store=os.path.join(destination_path, ZARR_PAT_ARCHIVE), mode="a")
        pat = root.create_group(name=name, overwrite=True)
        for k, v in values.items():
            arr_len = len(v.shape)
            if arr_len > 2:
                chunks = D3_PAT_CHUNKS
            else:
                chunks = PAT_CHUNKS
            pat.array(name=k, data=v, chunks=chunks)

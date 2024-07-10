import os
from typing import List

import numpy as np

from .data_loader_interface import DataLoaderInterface
from .. import CubeLoaderInterface, AnnotationMaskLoaderInterface
from ...data_storage import DataStorage
from ...paths.path_sort import folder_sort
from provider import get_data_storage

from configuration.parameter import (
    DICT_ORIGINAL_NAME,
    DICT_y,
    STORAGE_TYPE,
)


class DataLoaderFolder(DataLoaderInterface):
    def __init__(self, config, cube_loader: CubeLoaderInterface, mask_loader: AnnotationMaskLoaderInterface,
                 data_storage: DataStorage):
        super().__init__(config, cube_loader, mask_loader, data_storage)

    @classmethod
    def read_and_save(cls, destination_path: str, config, paths: str | List[str]):
        print(f"Read data for patient {paths[0]}", flush=True)
        data_storage = get_data_storage(typ=STORAGE_TYPE)
        first = True
        for path in paths[1]:
            print(f"Reading {path}", flush=True)
            name, values = cls.read_data_task(cube_path=path,
                                              config=config)
            values[DICT_ORIGINAL_NAME] = np.array([name] * values[DICT_y].shape[0])

            if first:
                data_storage.save_group(save_path=destination_path,
                                        group_name=paths[0],
                                        datas=values)
                first = False
            else:
                data_storage.append_data(file_path=os.path.join(destination_path, paths[0]),
                                         append_datas=values)

    def _paths_to_load(self, root_path: str) -> List[List[str | List[str]]]:
        name_and_paths = folder_sort(paths=self._get_raw_paths(root_path=root_path))
        return [[name, paths] for name, paths in name_and_paths.items()]

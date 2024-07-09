from typing import List

from .data_loader_interface import DataLoaderInterface
from .. import CubeLoaderInterface, AnnotationMaskLoaderInterface
from ...data_storage import DataStorage
from provider import get_data_storage


class DataLoaderBase(DataLoaderInterface):
    def __init__(self, config, cube_loader: CubeLoaderInterface, mask_loader: AnnotationMaskLoaderInterface,
                 data_storage: DataStorage):
        super().__init__(config, cube_loader, mask_loader, data_storage)

    @classmethod
    def read_and_save(cls, destination_path: str, paths: str | List[str]):
        # TODO: Is not the best solution to import, but can't give a module to a pool.map function
        import configuration.get_config as config
        print(f'Reading {paths}')
        name, values = cls.read_data_task(cube_path=paths,
                                          config=config)
        data_storage = get_data_storage(typ="npz")
        data_storage.save_group(save_path=destination_path,
                                group_name=name,
                                datas=values)

    def _paths_to_load(self, root_path: str) -> List[str]:
        return self._get_raw_paths(root_path=root_path)

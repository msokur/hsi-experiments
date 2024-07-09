import abc

import numpy as np
import os

from configuration.keys import (
    DataLoaderKeys as DLK,
)


class CubeLoaderInterface:
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def get_cube(self, cube_path: str) -> np.ndarray:
        pass

    def get_cube_name(self, cube_path: str) -> str:
        return os.path.split(p=cube_path)[-1].split(".")[0].split(self.config.CONFIG_DATALOADER[DLK.NAME_SPLIT])[0]

    @staticmethod
    @abc.abstractmethod
    def get_extension() -> str:
        pass

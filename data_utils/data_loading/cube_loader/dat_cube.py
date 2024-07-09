import numpy as np

from . import CubeLoaderInterface
from data_utils.hypercube_data import HyperCube

from configuration.keys import (
    DataLoaderKeys as DLK,
)


class DatCube(CubeLoaderInterface):
    def __init__(self, config):
        super().__init__(config)

    def get_cube(self, cube_path: str) -> np.ndarray:
        """ Load the HSI-cube from a .at file

        :param cube_path: Path to .dat file

        :return: Numpy array with the spectrum
        """
        spectrum = HyperCube(address=cube_path).cube_matrix(first_nm=self.config.CONFIG_DATALOADER[DLK.FIRST_NM],
                                                            last_nm=self.config.CONFIG_DATALOADER[DLK.LAST_NM])

        return spectrum

    @staticmethod
    def get_extension() -> str:
        return ".dat"

import numpy as np
import scipy.io as sio

from . import CubeLoaderInterface

from configuration.keys import (
    DataLoaderKeys as DLK
)


class MatCube(CubeLoaderInterface):
    def __init__(self, config):
        super().__init__(config)

    def get_cube(self, cube_path: str) -> np.ndarray:
        """ Load the HSI-cube from a .mat file

        :param cube_path: Path to .mat file

        :return: Numpy array with the spectrum
        """
        data = sio.loadmat(cube_path)
        spectrum = data[self.config.CONFIG_DATALOADER[DLK.SPECTRUM]]

        return spectrum

    @staticmethod
    def get_extension() -> str:
        return ".mat"
